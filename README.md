<div align="center">
<h1 align="center">ObjFormer</h1>

<h3>Learning Land-Cover Changes From Paired Map Data and Optical Imagery via Object-Guided Transformer
</h3>

[Hongruixuan Chen](https://scholar.google.ch/citations?user=XOk4Cf0AAAAJ&hl=zh-CN&oi=ao)<sup>1,3</sup>, [Cuiling Lan](https://scholar.google.com/citations?user=XZugqiwAAAAJ&hl=zh-CN)<sup>2</sup>, [Jian Song](https://scholar.google.ch/citations?user=CgcMFJsAAAAJ&hl=zh-CN)<sup>1,3</sup>, [Clifford Broni-Bediako](https://scholar.google.co.jp/citations?user=Ng45cnYAAAAJ&hl=en)<sup>3</sup>, [Junshi Xia](https://scholar.google.com/citations?user=n1aKdTkAAAAJ&hl=en)<sup>3</sup>, [Naoto Yokoya](https://scholar.google.co.jp/citations?user=DJ2KOn8AAAAJ&hl=en)<sup>1,3 *</sup>

[![TGRS paper](https://img.shields.io/badge/TGRS-paper-00629B.svg)](https://ieeexplore.ieee.org/abstract/document/10551264)  [![arXiv paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2310.02674)  [![Zenodo Datasets](https://img.shields.io/badge/Zenodo-Datasets-green)](https://zenodo.org/records/14028095) 


<sup>1</sup> The University of Tokyo, <sup>2</sup> Microsoft Research Asia,  <sup>3</sup> RIKEN AIP,  <sup>*</sup> Corresponding author

[**Overview**](#overview) | [**Get Started**](#%EF%B8%8Flets-get-started) | [**Taken Away**](#%EF%B8%8Fresults-taken-away) | [**Common Issues**](#common-issues) | [**Others**](#q--a) 

</div>

## 🛎️Updates
* **` Notice`**: ObjFormer has been accepted by [IEEE TGRS](https://ieeexplore.ieee.org/document/10551264)! We will upload the dataset and code soon. We'd appreciate it if you could give this repo a ⭐️**star**⭐️ and stay tuned!!
* **` Nov. 04nd, 2024`**: We have updated the code for benchmark, including code for some of the models as well as training and evaluation scripts. You are welcome to download and use them!
* **` July 01st, 2024`**: We have uploaded [OpenMapCD dataset](https://zenodo.org/records/14028095). You are welcome to download and use it!


## 🔭Overview

* [**OpenMapCD**](https://zenodo.org/records/14028095) is the first benchmark dataset for multimodal change detecton tasks on optical remote sensing imagery and map data, with 1,287 samples from 40 regions across six continents, supoorting both binary and semantic change detection. 


* [**ObjFormer**](https://ieeexplore.ieee.org/document/10565926) serves as a robust and efficient benchmark for the proposed multimodal change detection tasks by combining OBIA techniques with self-attention mechanisms.

<img src="./fig/Overall_framework.jpg">


## 📋To Do List

- [ ] Release the ObjFormer code
- [x] Release the benchmark training and evalution code
- [x] Release the OpenMapCD dataset
- [x] Release the ObjFormer & OpenMapCD paper

## 🗝️Let's Get Started!
### `A. Installation`
**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/ChenHongruixuan/ObjFormer.git
cd ObjFormer
```

**Step 2: Environment Setup:**

It is recommended to set up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n objformer
conda activate objformer
```

***Install dependencies***

```bash
pip install -r requirements.txt
```

### `B. Data Preparation`

Download OpenMapCD dataset from [Zenodo](https://zenodo.org/records/14028095) and put it under the [dataset] folder. It will have the following structure: 
```
${DATASET_ROOT}   # Dataset root directory
├── Benchmark
│   │
│   ├── OPT     # Optical remote sensing images
│   │   ├──aachen_1.png
│   │   ├──aachen_2.png
│   │   ...
│   │
│   ├── OSM     # OpenStreetMap data
│   │   ├ ... 
│   │
│   ├── LC_GT_OPT  # Land-cover labels of optical images	
│   │   ├ ... 
│   │     
│   ├── LC_GT_OSM  # Land-cover labels of OSM data
│   │   ├ ... 
│   │     
│   ├── BC_GT   # Binary change detection labels
│   │   ├ ... 
│   │
│   ├── SC_GT   # Semantic change detection labels
│   │   ├ ... 
│   │  
│   ├── train_list.txt   # Data name list, recording all the names of training data
│   └── test_list.txt    # Data name list, recording all the names of testing data  
│   
└── Application
    ├── ...
    ...
```

### `C. Model Training`

***Binary change detection***

The following commands show how to train and evaluate the benchmark model on the OpenMapCD dataset for binary change detection:
```bash
python script/train_benchmark_bcd.py --dataset_path '<your project path>/dataset/OpenMapCD/benchmark' \
                                     --batch_size 16 \
                                     --crop_size 512 \
                                     --max_iters 7500 \
                                     --train_data_list_path '<your project path>/dataset/OpenMapCD/benchmark/train_list.txt' \
                                     --eval_data_list_path '<your project path>/dataset/OpenMapCD/benchmark/eval_list.txt' \
                                     --model_param_path '<your project path>saved_weight' \
                                     --learning_rate 1e-4 \
                                     --weight_decay 5e-3 \
                                     --model_type 'FCEF' 
```


***Semantic change detection***

The following commands show how to train and evaluate the benchmark model on the OpenMapCD dataset for semantic change detection:
```bash
python script/train_benchmark_scd.py  --dataset_path '<your project path>/dataset/OpenMapCD/benchmark' \
                                      --batch_size 16 \
                                      --crop_size 512 \
                                      --max_iters 10000 \
                                      --train_data_list_path '<your project path>/dataset/OpenMapCD/benchmark/train_list.txt' \
                                      --eval_data_list_path '<your project path>/dataset/OpenMapCD/benchmark/eval_list.txt' \
                                      --model_param_path '<your project path>/saved_weight' \
                                      --learning_rate 1e-4 \
                                      --weight_decay 5e-3 \
                                      --model_type 'HRSCD-S4_RCE' 
```


## 📜Reference

If this code or dataset contributes to your research, please kindly consider citing our paper and give this repo ⭐️ :)
```
@ARTICLE{Chen2024ObjFormer,
  author={Chen, Hongruixuan and Lan, Cuiling and Song, Jian and Broni-Bediako, Clifford and Xia, Junshi and Yokoya, Naoto},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ObjFormer: Learning Land-Cover Changes From Paired OSM Data and Optical High-Resolution Imagery via Object-Guided Transformer}, 
  year={2024},
  volume={62},
  number={},
  pages={1-22},
  doi={10.1109/TGRS.2024.3410389}
}
```


## 🔗Other links
If you are interested in land-cover mapping and domain adaptation in remote sensing using synthetic datasets, you can also follow our three datasets below.

* *[OpenEarthMap dataset](https://open-earth-map.org/): a benchmark dataset for global sub-meter level land cover mapping.*

* *[SyntheWorld dataset](https://github.com/JTRNEO/SyntheWorld): a large-scale synthetic remote sensing datasets for land cover mapping and building change detection.* 

* *[SynRS3D dataset](https://github.com/JTRNEO/SyntheWorld): a large-scale synthetic remote sensing datasets for global 3D semantic uploadnderstanding.* 


## 🙋Q & A
**For any questions, please [contact us.](mailto:Qschrx@gmail.com)**
