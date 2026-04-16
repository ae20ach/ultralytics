# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from functools import partial, wraps
from pathlib import Path

import torch

from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
from .track_tracker import TRACKTRACK

# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT, "tracktrack": TRACKTRACK}


def _compute_dets_del(predictor, raw_preds):
    """Compute TrackTrack's D_del set from raw detector predictions.

    D_del are high-confidence detections that were removed by the tight NMS but survive a looser
    NMS pass. TrackTrack uses these as additional low-priority candidates during association (paper
    Eq. 1).

    Args:
        predictor (object): Predictor instance exposing `args`, `results`, and the cached
            `_preproc_img_shape`.
        raw_preds (torch.Tensor): Raw predictions captured before NMS by the postprocess wrapper.

    Returns:
        (list[tuple | None]): Per-batch `(xywh, conf, cls)` bundles, or None when no deleted
            detections are found for that batch element.
    """
    from torchvision.ops import box_iou

    from ultralytics.utils import nms, ops

    is_obb = predictor.args.task == "obb"
    preds_loose = nms.non_max_suppression(
        raw_preds,
        predictor.args.conf,
        0.95,  # loose IoU; paper uses 0.95 on top of the tight 0.80 pass
        predictor.args.classes,
        predictor.args.agnostic_nms,
        max_det=predictor.args.max_det,
        nc=0 if predictor.args.task == "detect" else len(predictor.model.names),
        end2end=getattr(predictor.model, "end2end", False),
        rotated=is_obb,
    )

    dets_del_list = []
    im_shape = getattr(predictor, "_preproc_img_shape", None)
    for loose, result in zip(preds_loose, predictor.results):
        det_boxes = (result.obb if is_obb else result.boxes).cpu()
        if len(loose) == 0 or len(det_boxes) == 0:
            dets_del_list.append(None)
            continue

        # Scale the loose-NMS boxes from preprocessed space back to the original image
        loose_scaled = loose.clone()
        if im_shape is not None:
            loose_scaled[:, :4] = ops.scale_boxes(im_shape, loose_scaled[:, :4], result.orig_shape)

        tight_xyxy = det_boxes.xyxy.cpu()
        loose_xyxy = loose_scaled[:, :4].cpu()
        if tight_xyxy.numel() == 0 or loose_xyxy.numel() == 0:
            dets_del_list.append(None)
            continue

        # A loose-NMS detection counts as D_del if it does not overlap any tight-NMS detection
        ious = box_iou(loose_xyxy, tight_xyxy)
        max_iou, _ = ious.max(dim=1)
        del_mask = max_iou < 0.97
        if not del_mask.any():
            dets_del_list.append(None)
            continue

        del_boxes = loose_scaled[del_mask]
        del_xywh = ops.xyxy2xywh(del_boxes[:, :4])
        dets_del_list.append((del_xywh.numpy(), del_boxes[:, 4].numpy(), del_boxes[:, 5].numpy()))

    return dets_del_list


def on_predict_start(predictor: object, persist: bool = False) -> None:
    """Initialize trackers for object tracking during prediction.

    Args:
        predictor (ultralytics.engine.predictor.BasePredictor): The predictor object to initialize
            trackers for.
        persist (bool, optional): Whether to reuse existing trackers if they are already attached.

    Examples:
        Initialize trackers for a predictor object
        >>> predictor = SomePredictorClass()
        >>> on_predict_start(predictor, persist=True)
    """
    if predictor.args.task == "classify":
        raise ValueError("❌ Classification doesn't support 'mode=track'")

    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**YAML.load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort", "tracktrack"}:
        raise AssertionError(
            f"Only 'bytetrack', 'botsort', and 'tracktrack' are supported for now, but got '{cfg.tracker_type}'"
        )

    predictor._feats = None  # reset in case used earlier
    if hasattr(predictor, "_hook"):
        predictor._hook.remove()
    if cfg.tracker_type in {"botsort", "tracktrack"} and cfg.with_reid and cfg.model == "auto":
        from ultralytics.nn.modules.head import Detect

        if not (
            isinstance(predictor.model.model, torch.nn.Module)
            and isinstance(predictor.model.model.model[-1], Detect)
            and not predictor.model.model.model[-1].end2end
        ):
            cfg.model = "yolo26n-cls.pt"
        else:
            # Register hook to extract input of Detect layer
            def pre_hook(module, input):
                predictor._feats = list(input[0])  # unroll to new list to avoid mutation in forward

            predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video

    # For TrackTrack: wrap postprocess to capture raw preds for D_del computation
    if cfg.tracker_type == "tracktrack" and not hasattr(predictor, "_orig_postprocess"):
        orig_postprocess = predictor.postprocess

        @wraps(orig_postprocess)
        def _postprocess_with_raw_preds(preds, img, *args, **kwargs):
            predictor._raw_preds = preds.clone() if isinstance(preds, torch.Tensor) else preds
            predictor._preproc_img_shape = img.shape[2:]  # (H, W) of the preprocessed image
            return orig_postprocess(preds, img, *args, **kwargs)

        predictor._orig_postprocess = orig_postprocess
        predictor.postprocess = _postprocess_with_raw_preds


def on_predict_postprocess_end(predictor: object, persist: bool = False) -> None:
    """Postprocess detected boxes and update with object tracking.

    Args:
        predictor (object): The predictor object containing the predictions.
        persist (bool, optional): Whether to persist the trackers if they already exist.

    Examples:
        Postprocess predictions and update with tracking
        >>> predictor = YourPredictorClass()
        >>> on_predict_postprocess_end(predictor, persist=True)
    """
    is_obb = predictor.args.task == "obb"
    is_stream = predictor.dataset.mode == "stream"

    # Compute D_del for TrackTrack if raw preds are available
    dets_del_list = None
    raw_preds = getattr(predictor, "_raw_preds", None)
    if raw_preds is not None and isinstance(predictor.trackers[0], TRACKTRACK):
        dets_del_list = _compute_dets_del(predictor, raw_preds)
        predictor._raw_preds = None  # clear

    for i, result in enumerate(predictor.results):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(result.path).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = (result.obb if is_obb else result.boxes).cpu().numpy()
        if isinstance(tracker, TRACKTRACK):
            dets_del = dets_del_list[i] if dets_del_list is not None else None
            tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None), dets_del=dets_del)
        else:
            tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None))
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = result[idx]

        update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def register_tracker(model: object, persist: bool) -> None:
    """Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.

    Examples:
        Register tracking callbacks to a YOLO model
        >>> model = YOLOModel()
        >>> register_tracker(model, persist=True)
    """
    model.add_callback("on_predict_start", partial(on_predict_start, persist=persist))
    model.add_callback("on_predict_postprocess_end", partial(on_predict_postprocess_end, persist=persist))
