from ultralytics import YOLOE,YOLO


import os


import argparse
argparser=argparse.ArgumentParser(description="merge visual prompt and seg head for yoloe")
argparser.add_argument("--scale", type=str, default="26s-seg")
argparser.add_argument("--vp_weight", type=str, default="yoloe26s-vp.pt")
argparser.add_argument("--seg_weight", type=str, default="yoloe-26s-seg-det.pt")
argparser.add_argument("--im", type=str, default="ultralytics/assets/bus.jpg")
argparser.add_argument("--save_dir", type=str, default="runs/fuse_seg_vp")
args=argparser.parse_args() 

scale=args.scale
vp_weight=args.vp_weight
seg_weight=args.seg_weight
im=args.im
save_dir=args.save_dir  


def test_yoloe26_vp_func(model,res_im_path):

    end2end=True

    # import os
    # if not os.path.exists(model):
    #     raise FileNotFoundError(f"Please download the file '{model}' and place it in the 'weights' directory.")

    import numpy as np

    visual_prompts = dict(
        bboxes=np.array(
            [
                [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
                [120, 425, 160, 445],  # Box enclosing glasses
            ],
        ),
        cls=np.array(
            [
                0,  # ID to be assigned for person
                1,  # ID to be assigned for glasses
            ]
        ),
    )

    im="../ultralytics/ultralytics/assets/bus.jpg"

    from ultralytics import YOLO,YOLOE



    if not end2end:
        del model.model.model[-1].one2one_cv2
        del model.model.model[-1].one2one_cv3
        del model.model.model[-1].one2one_cv4
        model.model.end2end = False


    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor, YOLOEVPDetectPredictor
    results = model.predict(
        im,
        visual_prompts=visual_prompts,
        # predictor=YOLOEVPDetectPredictor,
        predictor=YOLOEVPSegPredictor,
        conf=0.01,
    )[0]


    import os
    os.makedirs(os.path.dirname(res_im_path), exist_ok=True)
    print(f"Image saved to {os.path.abspath(res_im_path)}")
    results.save(res_im_path)



# # good style
# model=YOLOE(seg_weight)
# model_vp=YOLOE(vp_weight)
# model.model.model[-1].savpe = model_vp.model.model[-1].savpe
# model.save(f"weights/yoloe-{scale}-ye.pt")


model=YOLO(f"yoloe-{scale}.yaml")
model.load(seg_weight)
from ultralytics.utils.torch_utils import strip_optimizer
strip_optimizer(vp_weight)
model.load(vp_weight)


model.args['clip_weight_name']="mobileclip2:b"
test_yoloe26_vp_func(model,f"./runs/temp/img1.jpg")




model.save(f"weights/yoloe-{scale}-ye.pt")
model=YOLOE(f"weights/yoloe-{scale}-ye.pt")
model.args['clip_weight_name']="mobileclip2:b"
test_yoloe26_vp_func(model,f"./runs/temp/img2.jpg")




###########################################################


"""



 # 26n seg+vp fusion infer test
    python runs/tools/fuse_seg_vp_ye.py --scale 26n-seg \
    --seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26n-seg_ptwbest_tp_bs256_epo10_close2_engine_yoloe_ye_data.yaml_seg-ultra6/weights/best.pt \
     --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26n_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra6\]/weights/best.pt \
    --im ultralytics/assets/bus.jpg \
     --save_dir runs/merge_vp_seg/


# 26s seg+vp fusion infer test
 python runs/tools/fuse_seg_vp_ye.py --scale 26s-seg \
 --seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26s-seg_ptwbest_tp_bs256_epo10_close2_engine_yoloe_ye_data.yaml_seg-ultra6/weights/best.pt \
  --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26s_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra6\]/weights/best.pt \
 --im ultralytics/assets/bus.jpg \
 --save_dir runs/merge_vp_seg/




# 26m seg+vp fusion infer test
 python runs/tools/fuse_seg_vp_ye.py --scale 26m-seg \
 --seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26m-seg_ptwbest_tp_bs256_epo10_close2_engine_ye_data_seg-ultra7/weights/best.pt \
  --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26m_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra7\]/weights/best.pt \
 --im ultralytics/assets/bus.jpg \
 --save_dir runs/merge_vp_seg/

 # 26l seg+vp fusion infer test
python runs/tools/fuse_seg_vp_ye.py --scale 26l-seg \
--seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26l-seg_ptwbest_tp_bs256_epo10_close2_engine_yoloe_ye_data.yaml_seg-ultra6/weights/best.pt \
    --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26l_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra6\]/weights/best.pt \
--im ultralytics/assets/bus.jpg \
    --save_dir runs/merge_vp_seg/


# 26x seg+vp fusion infer test
 python runs/tools/fuse_seg_vp_ye.py --scale 26x-seg \
 --seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26x-seg_ptwbest_tp_bs256_epo10_close2_engine_yoloe_ye_data.yaml_seg-ultra4/weights/best.pt \
  --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26x_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra7\]/weights/best.pt \
 --im ultralytics/assets/bus.jpg \
 --save_dir runs/merge_vp_seg/  

"""