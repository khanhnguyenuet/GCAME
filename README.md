# GCAME: Gaussian Class Activation Mapping Explainer
This repo is the official implementation of GCAME.

-----------------------------------------
GCAME is a CAM-based XAI method which used Gaussian mask and activation map to explain the prediction of Object Detector. It is able to highlight the regions belonging to the target object and can run in very short time.

Authors:
<br>
Quoc Khanh Nguyen, Truong Thanh Hung Nguyen, Vo Thanh Khang Nguyen, Van Binh Truong, Quoc Hung Cao

## Method
<div align="center">
  <img src="https://raw.githubusercontent.com/khanhnguyenuet/GCAME/main/figures/method.png">
</div>
<p align="center">
  Figure 1: Pipeline of GCAME.
</p>

## Target layers
- Faster-RCNN:
```
    backbone.fpn.inner_blocks.0.0,
    backbone.fpn.layer_blocks.0.0,

    backbone.fpn.inner_blocks.1.0,
    backbone.fpn.layer_blocks.1.0,

    backbone.fpn.inner_blocks.2.0,
    backbone.fpn.layer_blocks.2.0,

    backbone.fpn.inner_blocks.3.0,
    backbone.fpn.layer_blocks.3.0,
```

- YOLOX-l:
```
    head.cls_convs.0.0.act
    head.cls_convs.0.1.act,

    head.cls_convs.1.0.act,
    head.cls_convs.1.1.act,

    head.cls_convs.2.0.act,
    head.cls_convs.2.1.act,
```

## Results
<div align="center">
  <img src="https://raw.githubusercontent.com/khanhnguyenuet/GCAME/main/figures/method.png">
</div>
<p align="center">
  Figure 1: Pipeline of GCAME.
</p>