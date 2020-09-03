# DESC: Domain Adaptation for Depth Estimation via Semantic Consistency
This is the PyTorch implementation for our BMVC20 (Oral) paper:

**A. Lopez-Rodriguez, K. Mikolajczyk. DESC: Domain Adaptation for Depth Estimation via Semantic Consistency. [Paper](https://www.bmvc2020-conference.com/assets/papers/0122.pdf)**


## Environment/Requirements
Tested with Pytorch 1.4/1.5, CUDA 10.1, Ubuntu 18.04 and Python 3.6.9.

You need to install the Detectron2 library (used for semantic information) following [these instructions](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). The pretrained panoptic segmentation model can be downloaded from [here](https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl).

You also need to install OpenCV, ImageIO and SciPY, which can be done using:

`pip install -r requirements.txt`
 
## Datasets
[KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)

We provide the KITTI ground-truth depth maps for the eigen test split [here](https://imperialcollegelondon.box.com/s/l94jgky0i30mx3vbblk43absl5f9jv3c) in the file `gt_depths.npz`, which are generated using the `export_gt_depth.py` in the [Monodepth2](https://github.com/nianticlabs/monodepth2) repository.

[vKITTI](https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/)


## Training (1080 Ti, 11GB)
We first train the networks separately by running the following two scripts

`./pretrain_depth.sh VKITTI_ROOT_FOLDER KITTI_ROOT_FOLDER`

`./pretrain_semantic_depth.sh VKITTI_ROOT_FOLDER KITTI_ROOT_FOLDER`

We then train them jointly to get our final model by using

`./joint_training.sh  VKITTI_ROOT_FOLDER KITTI_ROOT_FOLDER`

## Test
Pretrained models for the depth estimation network can be found in [this link](https://imperialcollegelondon.box.com/s/l94jgky0i30mx3vbblk43absl5f9jv3c). You need to have the ground-truth for the test data in the root folder, which is also given in the [same link](https://imperialcollegelondon.box.com/s/l94jgky0i30mx3vbblk43absl5f9jv3c) in `gt_depths.npz` as mentioned in the Datasets section.

To test the models we can run the following command

`./test.sh KITTI_ROOT_FOLDER`

By default it will load the model generated after finishing training, i.e, after running `./joint_training.sh`. You can modify test.py to load the pretrained models, we give examples to do so in the commented lines. Also, if you are evaluating the stereo-trained model, set the `disable_median_scaling` option in `evaluate_model` to 1.


## Citation
If you use DESC for your research, you can cite the paper using the following Bibtex entry:
```
@inproceedings{lopez2020desc,
  title={DESC: Domain Adaptation for Depth Estimation via Semantic Consistency},
  author={Lopez-Rodriguez, Adrian and Mikolajczyk, Krystian},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2020}
}
```

## Observations
Our reported results for GASDA are better than those in the original [GASDA](https://arxiv.org/abs/1904.01870) paper due to an indexing bug the original GASDA code. The indexing bug was related to the test ground-truth generation from the Velodyne data, which has already been fixed in GASDA and now their results match those reported in our paper.

## Acknowledgments
Code is inspired by [T^2Net](https://github.com/lyndonzheng/Synthetic2Realistic) and [GASDA](https://github.com/sshan-zhao/GASDA).

## Contact
Adrian Lopez-Rodriguez: al4415@ic.ac.uk
