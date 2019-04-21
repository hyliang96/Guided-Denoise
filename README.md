# Guided-Denoise

The winning submission for NIPS 2017: Defense Against Adversarial Attack of team TSAIL

# Paper 

[Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser](https://arxiv.org/abs/1712.02976)

# Our Experiment environment

tensorflow-gpu=1.9.0

GeForce GTX TITAN X

# File Description

The files and dirs directly under this repo folder are the following:

### dataset

- prepare_data.ipynb: generate dataset
- Originset, Originset_test: original (unadversarial) images
- Advset: the adversarial images

### model code

- Attackset: attacker models
- nips_deploy: classifier models
- Exps: the defenser model

### checkpoints

- checkpoints: the checkpoints of, download [here](https://pan.baidu.com/s/1kVzP9nL)
  - target classification models for the attacks used to generate training and testing exsamples
  - baseline defence models with adversarial training

### scripts

- toolkit: the program running the attack in batch
- GD_train: train the defense model using guided denoise
- PD_train: train the defense model using pixel denoise

# Dataset

You need to download ImageNet before preprocessing dataset , and 

* please make the ImageNet training images in 1000 folders, a folder for each class, like:

  `Path_to_ImageNet_train/n04485082/n04485082_1000.JPEG`

* please make the ImageNet validating images in one folder, like:

  `Path_to_ImageNet_val_data//ILSVRC2012_val_00033334.JPEG	`

# How to use

### Preprocess dataset

<!--we do this on gpuserver9:/home/haoyu/project/lab_project_handin2019/Guided-Denoise-->

<!--Path_to_ImageNet_train='/raid/tianyu/adv_train/imagenet_data/train'-->

<!--Path_to_ImageNet_val_data='/mfs/you/Imagenet/val_data/'-->

```bash
mkdir Ogirinset Originset_test # used to save images that prepare_data.ipynb will use 
```

Run `prepare_data.ipynb` with jupyter notebook, to convert ImageNet into `Ogirinset`, `Originset_test`

### Run Attackers

<!--we do this on gpuserver3:/home/haoyu/project/Guided-Denoise-->

To generate the attacking samples which will be used to train and test denoiser

```bash
mkdir Advset # used to save attacking samples
```

#### Generate training exsamples

- modifying arguments in `toolkit/run_attacks.sh`:

  - `--models=<attack_name1>,<attack_name2>,...`：<attack_namei> is the name of a dir under `Attackset/`
  - `--models=all`: all attacks in `Attackset/`

- run attacker

  ```bash
  bash toolkit/run_attacks.sh <gpuids> [other args]
  ```

  - `<gpuids>`: gpuids to run attack, numbers separated by comma. This step only consumes one GPU.

#### Generate testing exsamples

- change `DATASET_DIR`

  in `toolkit/run_attacks.sh`, find  `DATASET_DIR="${parentdir}/Originset"`,  change `Originset` to `Originset_test`

- rerun attacker

  ```bash
  bash toolkit/run_attacks.sh <gpuids>
  ```

### Run guided denoiser

- run  guided denoiser

  ```bash
  python GD_train/main.py --exp <defense_model> [other_arguments]
  ```

  - `<defense_model>`=` sample`: to use defense model in  `Exps/sample/model.py`
  - you can also specify other arguements like `--xxxx xxxx` defined in  `GD_train/main.py`

