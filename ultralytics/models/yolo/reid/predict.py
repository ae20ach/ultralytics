# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import torch
from PIL import Image

from ultralytics.data.augment import classify_transforms
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class ReidPredictor(BasePredictor):
    """Predictor for person re-identification models.

    Extracts normalized feature embeddings from input images.

    Examples:
        >>> from ultralytics.models.yolo.reid import ReidPredictor
        >>> args = dict(model="yolo26n-reid.pt", source="path/to/query/")
        >>> predictor = ReidPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize ReidPredictor.

        Args:
            cfg (dict): Default configuration.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): Callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "reid"

    def setup_source(self, source):
        """Set up source and transforms."""
        super().setup_source(source)
        self.transforms = (
            self.model.model.transforms
            if self.model.format == "pt" and hasattr(self.model.model, "transforms")
            else classify_transforms(self.imgsz)
        )

    def preprocess(self, img):
        """Convert input images to model-compatible tensor format."""
        if not isinstance(img, torch.Tensor):
            img = torch.stack(
                [self.transforms(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in img], dim=0
            )
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()

    def postprocess(self, preds, img, orig_imgs):
        """Process predictions to return Results objects with embeddings.

        Args:
            preds: Model output (embedding tuple or tensor).
            img: Preprocessed input images.
            orig_imgs: Original images.

        Returns:
            (list[Results]): Results with embedding stored as probs field.
        """
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        emb = preds[0] if isinstance(preds, (list, tuple)) else preds
        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=e)
            for e, orig_img, img_path in zip(emb, orig_imgs, self.batch[0])
        ]
