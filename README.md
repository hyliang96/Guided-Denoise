# Guided-Denoise

The winning submission for NIPS 2017: Defense Against Adversarial Attack of team TSAIL

# Paper 

[Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser](https://arxiv.org/abs/1712.02976)

# Our Experiment environment

GeForce GTX TITAN X, 12212 MB

Python 3.6.7

```bash
pip insatll tensorflow-gpu=1.9.0 torch==0.4.1 torchvision==0.2.1
```

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
- `Exps/sample/inceptionv3_state.pth`: [download here](https://github.com/lfz/Guided-Denoise/blob/master/Exps/sample/inceptionv3_state.pth)

### scripts

- toolkit: the program running the attack in batch
- GD_train: train the defense model using guided denoise
- PD_train: train the defense model using pixel denoise

# Dataset

You need to download ImageNet before preprocessing dataset , and 

- please make the ImageNet training images in 1000 folders, a folder for each class, like:

  `Path_to_ImageNet_train/n04485082/n04485082_1000.JPEG`

- please make the ImageNet validating images in one folder, like:

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

This step only consumes one GPU. (we used GeForce GTX TITAN X, 12212 MB)

```bash
mkdir Advset # used to save attacking samples
```

#### Generate training exsamples

- modifying arguments in `toolkit/run_attacks.sh`:

  - `--models=<attack_method1>,<attack_method2>,...`

    To see all attack methods, just run

    ```bash
    ls Attackset/
    ```

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

This step only consumes four GPUs. (we used GeForce GTX TITAN X, 12212 MB)

- Select attack methods for training a guided denoiser	

  edit `Exps/sample/train_attack.txt`, write in the fellowing form. The following setting is to reproducing our [paper](https://arxiv.org/abs/1712.02976).

  ```txt
  fgsm_v3_random,
  fgsm_inresv2_random,
  fgsm_resv2_random,
  fgsm_v3_resv2_inresv2_random,
  Iter2_v3_resv2_inresv2_random,
  Iter4_v3_resv2_inresv2_random,
  Iter8_v3_resv2_inresv2_random
  ```

  please make sure all method in `train_attack` are used to generated training exmaples which are set before in `toolkit/run_attacks.sh`.

  And so `Exps/sample/test_attack.txt` does, for testing the guided denoiser. The following setting is to reproducing our [paper](https://arxiv.org/abs/1712.02976).

  ```txt
  fgsm_v3_random,
  Iter4_v3_resv2_inresv2_random,
  fgsm_v4_random,
  Iter4_v4_eps4,
  Iter4_v4_eps16
  ```

- run  guided denoiser

  ```bash
  cd GD_train
  python /main.py --exp <defense_model> [other_arguments]
  ```

  - `<defense_model>`=` sample`: to use defense model in  `Exps/sample/model.py`

  - if return rerror "out of cuda memmory", then add `--batch-size 16`

  - The checkpoints of  guided denoiser are saved in `Exps/resnet/results/<time_stamp>/xxx.ckpt`

    add  `--save-dir <name>`, here `<name>` takes place of `time_stamp`. You can set a `<name>` for each expriment

  - add `--resume <path_to_the_checkpoint_file_you_want_to_load>` to continue from this checkpoint

  foremore arguments, see the head of  `GD_train/main.py`

example: here is what we run

# Our Running Log

```bash
gpuid 0,1,2 python main.py --exp sample --batch-size 32 --save-dir debug
```

> ```
> 
> ```

