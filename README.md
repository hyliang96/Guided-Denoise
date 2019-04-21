# Guided-Denoise

The winning submission for NIPS 2017: Defense Against Adversarial Attack of team TSAIL

# Paper 

[Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser](https://arxiv.org/abs/1712.02976)

# File Description

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

# How to use

### Preprocess dataset

Run `prepare_data.ipynb` with jupyter notebook, to convert raw dataset (like ImageNet, TinyImageNet) `Ogirinset` and `Originset_test`, which are used to run attackers and train denoiser. 

### Run Attackers

To generate the attacking samples which will be used to train and test denoiser

#### Generate training exsamples

- modifying arguments in `toolkit/run_attacks.sh`:

  - `--models=<attack_name1>,<attack_name2>,...`：<attack_namei> is the name of a dir under `Attackset/`
  - `--models=all`: all attacks in `Attackset/`

- run attacker

  ```bash
  bash toolkit/run_attacks.sh <gpuids> [other args]
  ```

  - `<gpuids>`: gpuids to run attack, numbers separated by comma. 

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

