# RoReg: Pairwise Point Cloud Registration with Oriented Descriptors and Local Rotations

We present RoReg, a novel point cloud registration framework that fully exploits oriented descriptors and estimated local rotations in the whole registration pipeline. Previous methods mainly focus on extracting rotation-invariant descriptors for registration but unanimously neglect the orientations of descriptors. In this paper, we show that the oriented descriptors and the estimated local rotations are very useful in the whole registration pipeline, including feature description, feature detection, feature matching, and transformation estimation. Consequently, we design a novel oriented descriptor RoReg-Desc and apply RoReg-Desc to estimate the local rotations. Such estimated local rotations enable us to develop a rotation-guided detector, a rotation coherence matcher, and a one-shot-estimation RANSAC, all of which greatly improve the registration performance. Extensive experiments demonstrate that RoReg achieves state-of-the-art performance on the widely-used 3DMatch and 3DLoMatch datasets, and also generalizes well to the outdoor ETH dataset. In particular, we also provide in-depth analysis on each component of RoReg, validating the improvements brought by oriented descriptors and the estimated local rotations.

## News

- 2022.9.16 The test code of RoReg is released. The paper is under review and we will release the whole code and supplementary materials as soon as the paper gets accepted.

## Performance

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes'>
  <td width=81 valign=top style='width:61.0pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>Datasets</span></p>
  </td>
  <td width=241 colspan=5 valign=top style='width:180.4pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>3DMatch</span></p>
  </td>
  <td width=231 colspan=5 valign=top style='width:173.4pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='tab-stops:58.8pt'><span lang=EN-US><span
  style='mso-tab-count:1'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span>3DLoMatch</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1'>
  <td width=81 valign=top style='width:61.0pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>#<span class=SpellE>Keypoints</span></span></p>
  </td>
  <td width=49 valign=top style='width:36.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>5000</span></p>
  </td>
  <td width=49 valign=top style='width:36.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>2500</span></p>
  </td>
  <td width=49 valign=top style='width:36.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>1000</span></p>
  </td>
  <td width=47 valign=top style='width:35.35pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>500</span></p>
  </td>
  <td width=47 valign=top style='width:35.4pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>250</span></p>
  </td>
  <td width=48 valign=top style='width:35.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>5000<o:p></o:p></span></p>
  </td>
  <td width=47 valign=top style='width:35.05pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>2500<o:p></o:p></span></p>
  </td>
  <td width=46 valign=top style='width:34.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>1000<o:p></o:p></span></p>
  </td>
  <td width=46 valign=top style='width:34.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>500<o:p></o:p></span></p>
  </td>
  <td width=45 valign=top style='width:33.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>250<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2'>
  <td width=81 valign=top style='width:61.0pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>IR(%)</span></p>
  </td>
  <td width=49 valign=top style='width:36.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>81.6</span></p>
  </td>
  <td width=49 valign=top style='width:36.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>80.2</span></p>
  </td>
  <td width=49 valign=top style='width:36.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>75.1</span></p>
  </td>
  <td width=47 valign=top style='width:35.35pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>74.1</span></p>
  </td>
  <td width=47 valign=top style='width:35.4pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>75.2</span></p>
  </td>
  <td width=48 valign=top style='width:35.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>39.6</span></p>
  </td>
  <td width=47 valign=top style='width:35.05pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>39.6</span></p>
  </td>
  <td width=46 valign=top style='width:34.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>34.0</span></p>
  </td>
  <td width=46 valign=top style='width:34.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>31.9</span></p>
  </td>
  <td width=45 valign=top style='width:33.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>34.5</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;mso-yfti-lastrow:yes'>
  <td width=81 valign=top style='width:61.0pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>RR(%)</span></p>
  </td>
  <td width=49 valign=top style='width:36.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>92.9</span></p>
  </td>
  <td width=49 valign=top style='width:36.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>93.2</span></p>
  </td>
  <td width=49 valign=top style='width:36.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>92.7</span></p>
  </td>
  <td width=47 valign=top style='width:35.35pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>93.3</span></p>
  </td>
  <td width=47 valign=top style='width:35.4pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>91.2</span></p>
  </td>
  <td width=48 valign=top style='width:35.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>70.3</span></p>
  </td>
  <td width=47 valign=top style='width:35.05pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>71.2</span></p>
  </td>
  <td width=46 valign=top style='width:34.6pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>69.5</span></p>
  </td>
  <td width=46 valign=top style='width:34.2pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>67.9</span></p>
  </td>
  <td width=45 valign=top style='width:33.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>64.3</span></p>
  </td>
 </tr>
</table>

## Requirements
Here we offer the FCGF backbone RoReg. Thus FCGF requirements need to be met:
- Ubuntu 14.04 or higher
- CUDA 11.1 or higher
- Python v3.7 or higher
- Pytorch v1.6 or higher

Specifically, The code has been tested with:
- Ubuntu 16.04, CUDA 11.1, python 3.7.10, Pytorch 1.7.1, GeForce RTX 2080Ti.


## Installation
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

## Dataset & Pretrained model
The datasets and pretrained weights have been uploaded to Google Cloud:
- [3DMatch_train](https://drive.google.com/file/d/1mfnGL8pRvc6Rw6m6YnvNKdbpGxGJ081G/view?usp=sharing);
- [3DMatch/3DLomatch](https://drive.google.com/file/d/1UzGBPce5VspD2YIj7zWrrJYjsImSEc-5/view?usp=sharing);
- [ETH](https://drive.google.com/file/d/1hyurp5EOzvWGFB0kOl5Qylx1xGelpxaQ/view?usp=sharing);
- [Pretrained Weights](https://drive.google.com/file/d/1egPh8m41sAXeOfgxIFhvzl1r4QbR4YFX/view?usp=sharing). (Already added to the main branch.)

Also, all data above can be downloaded in [BaiduDisk](https://pan.baidu.com/s/17Rs9cTd1CVl9sPi9l2FL2Q)(Code:1pj6), where the checkpoints of RoReg are saved in ```YOHO/RoReg/checkpoints```.

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

## Train

Currently the repository contains only the evaluation code because the paper is under reviewing. The whole code will be avaliable soon. Part of training code, i.e., our descriptor RoReg-Desc and one-shot transformation estimation, has been pushed to [YOHO](https://github.com/HpWang-whu/YOHO).

We already offer the pretrained models in ```./checkpoints```.

## Demo

With the pretrained models, you can try RoReg with:
```
python demo.py
```

## Test on the 3DMatch and 3DLoMatch
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


## Generalize to the ETH dataset
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


## Related Projects
[YOHO](https://github.com/HpWang-whu/YOHO) is the early work of RoReg.

We sincerely thank the contributing projects:
- [EMVN](http://github.com/daniilidis-group/emvn) for the group details;
- [FCGF](https://github.com/chrischoy/FCGF) for the backbone;
- [3DMatch](https://github.com/andyzeng/3dmatch-toolbox) for the 3DMatch dataset;
- [Predator](https://github.com/overlappredator/OverlapPredator) for the 3DLoMatch dataset;
- [ETH](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration) for the ETH dataset;
- [PerfectMatch](https://github.com/zgojcic/3DSmoothNet) for organizing the 3DMatch and ETH dataset.




