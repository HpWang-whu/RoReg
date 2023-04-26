<h1 align="center"> <p>🤗 RoReg</p></h1>
<h3 align="center">
<a href="https://doi.org/10.1109/TPAMI.2023.3244951" target="_blank">RoReg: Pairwise Point Cloud Registration with Oriented Descriptors and Local Rotations</a>
</h3>

<h3 align="center">
IEEE TPAMI 2023
</h3>

<p align="center">
<a href="https://hpwang-whu.github.io/" target="_blank">Haiping Wang</a><sup>*,1</sup>, 
<a href="https://liuyuan-pal.github.io/" target="_blank">Yuan Liu</a><sup>*,2</sup>,
<a href="https://qingyonghu.github.io/" target="_blank">Qingyong Hu</a><sup>3</sup>,
<a href="http://www.cs.ox.ac.uk/people/bing.wang/" target="_blank">Bing Wang</a><sup>4</sup>,
<a href="" target="_blank">Jianguo Chen</a><sup>5</sup>,
<a href="https://dongzhenwhu.github.io/" target="_blank">Zhen Dong</a><sup>&dagger;,1</sup>, 
<a href="http://yulanguo.me/" target="_blank">Yulan Guo</a><sup>6</sup>,
<a href="https://www.cs.hku.hk/people/academic-staff/wenping" target="_blank">Wenping Wang</a><sup>7</sup>
<a href="https://scholar.google.com/citations?hl=zh-CN&user=DZsF2oIAAAAJ" target="_blank">Bisheng Yang</a><sup>&dagger;,1</sup> <br>
</p>

<p align="center">
<sup>1</sup>Wuhan University &nbsp;&nbsp; 
<sup>2</sup>The University of Hong Kong &nbsp;&nbsp; 
<sup>3</sup>University of Oxford &nbsp;&nbsp; <br>
<sup>4</sup>The Hong Kong Polytechnic University &nbsp;&nbsp; 
<sup>5</sup>DiDi Chuxing &nbsp;&nbsp;  <br>
<sup>6</sup>Sun Yat-sen University &nbsp;&nbsp; 
<sup>7</sup>Texas A&M University &nbsp;&nbsp; <br>
<sup>*</sup>The first two authors contribute equally. &nbsp;&nbsp; 
<sup>&dagger;</sup>Corresponding authors. &nbsp;&nbsp; 
</p>

We present RoReg, a novel point cloud registration framework that fully exploits oriented descriptors and estimated local rotations in the whole registration pipeline. Previous methods mainly focus on extracting rotation-invariant descriptors for registration but unanimously neglect the orientations of descriptors. In this paper, we show that the oriented descriptors and the estimated local rotations are very useful in the whole registration pipeline, including feature description, feature detection, feature matching, and transformation estimation. Consequently, we design a novel oriented descriptor RoReg-Desc and apply RoReg-Desc to estimate the local rotations. Such estimated local rotations enable us to develop a rotation-guided detector, a rotation coherence matcher, and a one-shot-estimation RANSAC, all of which greatly improve the registration performance. Extensive experiments demonstrate that RoReg achieves state-of-the-art performance on the widely-used 3DMatch and 3DLoMatch datasets, and also generalizes well to the outdoor ETH dataset. In particular, we also provide in-depth analysis on each component of RoReg, validating the improvements brought by oriented descriptors and the estimated local rotations.

## 🆕 News
- 2023-02-28: A multiview registration mehtod [SGHR](https://github.com/WHU-USI3DV/SGHR) is accepted by CVPR 2023! 🎉🎉
- 2023-02-05 RoReg has been accepted by IEEE TPAMI! 🎉🎉
  
  [[TPAMI]](https://doi.org/10.1109/TPAMI.2023.3244951)  (*Early Access. We will fix some typographical errors like Fig.2 in the final publication.)

  [[Project_Page]](https://hpwang-whu.github.io/RoReg/) 

  [[Supplementary_Material]](media/RoReg_Appendix.pdf)
- 2022-09-16 The code of RoReg is released.
- 2022-06-30 The early work of RoReg, a.k.a [YOHO](https://github.com/HpWang-whu/YOHO), is accepted by ACM MM 2022! 🎉🎉


## 💻 Requirements
Here we offer the FCGF backbone RoReg. Thus FCGF requirements need to be met:
- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher

Specifically, The code has been tested with:
- Ubuntu 16.04, CUDA 11.1, python 3.7.10, Pytorch 1.7.1, GeForce RTX 2080Ti.


## 🔧 Installation
- First, create the conda environment:
  ```
    conda create -n roreg python=3.7
    conda activate roreg
  ```
- Second, intall Pytorch. We have checked version 1.7.1 and other versions can be referred to [Official Set](https://pytorch.org/get-started/previous-versions/).
  ```
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
  ```
- Third, install [MinkowskiEngine](https://github.com/stanfordvl/MinkowskiEngine) v0.5 for FCGF feature extraction:
  ```
    cd utils/MinkowskiEngine
    conda install openblas-devel -c anaconda
    export CUDA_HOME=/usr/local/cuda-11.1 #We have checked cuda-11.1.
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
    cd ../..
  ```
- Fourth, install other packages, here we use 0.8.0.0 version [Open3d](http://www.open3d.org/) for Ubuntu 16.04:
  ```
  pip install -r requirements.txt
  ```

- Finally, compile the [CUDA based KNN searcher](https://github.com/vincentfpgarcia/kNN-CUDA):
  ```
  cd utils/knn_search
  export CUDA_HOME=/usr/local/cuda-11.1 #We have checked cuda-11.1.
  python setup.py build_ext --inplace
  cd ../..
  ```

## 💾 Dataset & Pretrained model
The datasets and pretrained weights have been uploaded to Google Cloud:
- [3dm_train_rot](https://drive.google.com/file/d/15wrOvrDST1gl7dTVwzE937TCQfse6Ihk/view?usp=sharing);
- [3DMatch/3DLomatch](https://drive.google.com/file/d/1UzGBPce5VspD2YIj7zWrrJYjsImSEc-5/view?usp=sharing);
- [ETH](https://drive.google.com/file/d/1hyurp5EOzvWGFB0kOl5Qylx1xGelpxaQ/view?usp=sharing);
- [Pretrained Weights](https://drive.google.com/file/d/1YZixgOA1MrmTsaFFCKxO_CdUHsat8Nzi/view?usp=sharing). (Already added to the main branch.)

Also, all data above can be downloaded in [BaiduDisk](https://pan.baidu.com/s/1xvnXNENCLvIZ4jjIZP6Kxg)(Code:b4zd), where the checkpoints of RoReg and 3dm_train_rot are saved in ```YOHO/RoReg```.

Datasets above contain the point clouds (.ply) and keypoints (.txt, 5000 per point cloud) files. Please place the data to ```./data/origin_data``` following the example data structure as:

```
data/
├── origin_data/
    ├── 3dmatch/
    	└── kitchen/
            ├── PointCloud/
            	├── cloud_bin_0.ply
            	├── gt.log
            	└── gt.info
            └── Keypoints/
            	└── cloud_bin_0Keypoints.txt
    ├── 3dmatch_train/
    └── ETH/
```

## 🚅 Train
To train RoReg-Desc and local rotation estimation (one-shot transformation estimation) with the FCGF backbone we offer, you can first prepare the trainset:
```
python trainset.py --component GF
```
and conduct training of the two components by:
```
python Train.py --component GF # for RoReg-Desc, requiring ~250G storage space.
python Train.py --component ET # for local rotations
```

After the training of RoReg-Desc and local rotation estimation, you can follow the commonds to train rotation-guided detector yourself:
```
python trainset.py --component RD
python Train.py --component RD
```

To train rotation coherence matcher yourself, you can follow the commonds of 
```
python trainset.py --component RM # require ~300G storage space.
python Train.py --component RM
```

All models will be saved in ```./checkpoints/FCGF```.

## 🔦 Demo

With the pretrained models, you can try RoReg with:
```
python demo.py
```

## ✏️ Test on the 3DMatch and 3DLoMatch
With the TestData downloaded above, the test on 3DMatch and 3DLoMatch can be done by
- Prepare the testset
```
python testset.py --dataset 3dmatch --voxel_size 0.025
```
- Eval the results:
```
python Test.py --RD --RM --ET yohoo --keynum 1000 --testset 3dmatch
```

```--RD``` denotes using the proposed rotation-guided detector, and we will use randomly-sampling without it.

```--RM``` denotes using the proposed rotation coherence matcher, and we will use NN+mutual without it.

```--ET``` denotes the choice of transformation estimation. ```yohoo``` means using the proposed one-shot transformation estimation, and we also offer another faster RANSAC variant ```yohoc```.

```--keynum``` denotes to sample how many keypoints in each scan for matching.

```--dataset``` denotes the evaluation dataset: ```3dmatch``` for the 3DMatch dataset and ```3dLomatch``` for the 3DLoMatch dataset.

More options as well as their descriptions can be found in ```Test.py```.

All the results will be placed to ```./data/YOHO_FCGF```.


## ✒️ Generalize to the ETH dataset
The generalization results on the outdoor ETH dataset can be got as follows:
- Prepare the testset
```
python testset.py --dataset ETH --voxel_size 0.15
```
- Eval the results:
```
python Test.py --RD --RM --ET yohoo --keynum 1000 --testset ETH --tau_2 0.2 --tau_3 0.5 --ransac_ird 0.5
```
All the results will be placed to ```./data/YOHO_FCGF```.


## 💡 Citation

If you find RoReg/YOHO useful in your research, please consider citing:

```
@inproceedings{wang2022you,
  title={You only hypothesize once: Point cloud registration with rotation-equivariant descriptors},
  author={Wang, Haiping and Liu, Yuan and Dong, Zhen and Wang, Wenping},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1630--1641},
  year={2022}

@ARTICLE{wang2023roreg,
  author={Wang, Haiping and Liu, Yuan and Hu, Qingyong and Wang, Bing and Chen, Jianguo and Dong, Zhen and Guo, Yulan and Wang, Wenping and Yang, Bisheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={RoReg: Pairwise Point Cloud Registration with Oriented Descriptors and Local Rotations}, 
  year={2023},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2023.3244951}}
}
```

## 🔗 Related Projects
Welcome to take a look at the homepage of our research group [WHU-USI3DV](https://github.com/WHU-USI3DV) ! We focus on 3D Computer Vision, particularly including 3D reconstruction, scene understanding, point cloud processing as well as their applications in intelligent transportation system, digital twin cities, urban sustainable development, and robotics.

[YOHO](https://github.com/HpWang-whu/YOHO) is the early work of RoReg and we sincerely thank the contributing projects:
- [EMVN](http://github.com/daniilidis-group/emvn) for the group details;
- [FCGF](https://github.com/chrischoy/FCGF) for the backbone;
- [3DMatch](https://github.com/andyzeng/3dmatch-toolbox) for the 3DMatch dataset;
- [Predator](https://github.com/overlappredator/OverlapPredator) for the 3DLoMatch dataset;
- [ETH](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration) for the ETH dataset;
- [PerfectMatch](https://github.com/zgojcic/3DSmoothNet) for organizing the 3DMatch and ETH dataset.