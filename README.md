# Fork of DSL used as a baseline

See the original readme below, it includes the installation instructions.

To run the code on our data, execute the following commands (substitutting relevant configs and paths):

```
conda activate feddgan

# train GAN
cd ~/s2/dsl/FedML/fedml_experiments/distributed/asdgan
sh run_asdgan_distributed_pytorch.sh 3 qata_gan_0.yml

# look at synthetic samples
python save_syn.py --cfg qata_gan_0.yml --batch_size 20 --save_dir ./run/qata/asdgan/experiment_0 --GPUid 0 --num_test 1 --epoch 200
# save synthetic samples
python save_syn.py --cfg qata_gan_0.yml --batch_size 20 --save_dir ./run/qata/asdgan/experiment_0 --GPUid 0 --num_test -1 --epoch 200 --save_data

# train a segmentation model
cd ~/s2/dsl/segmentation/
python train.py -d 0,1 -c config_files/qata_unet_0.json
# test
python test_general_2d.py -d 0,1 -c config_files/qata_unet_0.json -r saved/models/qata_unet_0/0314_095906/model_best.pth

```

# DSL
[![DOI](https://zenodo.org/badge/531561838.svg)](https://zenodo.org/badge/latestdoi/531561838)


Official implementation of paper "Mining Multi-Center Heterogeneous Medical Data with Distributed Synthetic Learning"

## System Requirements

- Hardware:
    - One Nvidia GPU with video memory 12GB+
- Software:
    - Linux(we use Ubuntu 18.04.5)
    - Python 3.6
    - pytorch 1.3.1+
    - CUDA10.0

## Installation

- install conda from [here](https://www.anaconda.com/products/individual)

- setup environments(~15min):

```
conda create --name feddgan --file requirements.txt
conda activate feddgan
```

- install FedML

http://doc.fedml.ai/#/installation-distributed-computing

## Data Preparation

Data split configurations are stored in FedDGAN/data_split_config
Download each datasets(GAN related) using the scripts: 

```
sh download_nuclei_dataset.sh
sh download_brats_dataset.sh
sh download_cardiac_dataset.sh
```

You can also download from the separate challenge webpages:

- The cardiac datasets include publicly available MM-WHS (https://zmiclab.github.io/projects/mmwhs/), ASOCA (https://asoca.grand-challenge.org/), and CAT08 (http://coronary.bigr.nl/centerlines/). The WHS masks for ASOCA and CAT08 data can be downloaded at https://rutgers.app.box.com/folder/147492331099.

- The brain data are available through BraTS 2018 Challenge  (https://www.med.upenn.edu/sbia/brats2018.html).

- The nuclei data are now part of the MoNuSeg training set (https://monuseg.grand-challenge.org/Data/).

## Train DSL on cardiac CTA, brain BraTS18, and Nuclei datasets(Stand-alone version)

- Run the commands under `DSL` folder.

- Train DSL on cardiac dataset.
```
python train.py --dataroot PATH_TO_YOUR_DATASET --name CTA3db_nature --input_nc 1 --output_nc 1 --model fsl3db --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode fsl3db --pool_size 0 --gpu_ids 1 --batch_size 10 --display_freq 300 --num_threads 0 --norm instance --d_size 3
```

- Train DSL on Brain MRI BraTS dataset
```
python train.py --dataroot PATH_TO_YOUR_DATASET --name Brats4ch3db_nature --input_nc 4 --output_nc 4 --model dadgan4ch --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode brats_4ch3db --pool_size 0 --gpu_ids 0 --batch_size 10 --num_threads 0 --norm instance --d_size 3
```

- Train DSL on Nuclei dataset
```
python train.py --dataroot PATH_TO_YOUR_DATASET --name hist_nature --model dadgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --delta_perceptual 10  --dataset_mode nuclei_split --pool_size 0  --gpu_ids 5 --batch_size 8 --norm instance --num_threads 0 --niter=200 --niter_decay=200 --d_size 4
```

#### Generate synthetic datasets

- Run the commands under `DSL` folder.

- Generate synthetic dataset on cardiac dataset

```
python save_syn_fsl.py --dataroot PATH_TO_YOUR_DATASET --name CTA3db_nature --input_nc 1 --output_nc 1 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode fsl1db --modality ct --epoch 200 --load_size 256 --crop_size 256 --gpu_ids 1 --results_dir results/CTA3db_nature
```

- Generate synthetic dataset on brain MRI dataset

```
python save_syn.py --dataroot PATH_TO_YOUR_DATASET --name Brats4ch3db_modality_nature --input_nc 4 --output_nc 4 --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode brats_4ch --epoch 160 --gpu_ids 1  --results_dir results/Brats4ch3db_modality_nature
```

- Generate synthetic dataset on Nuclei dataset

```
python save_syn_nuclei.py --dataroot PATH_TO_YOUR_DATASET  --name hist_norm --model pix2pix --netG resnet_9blocks --direction AtoB --dataset_mode nuclei --epoch 190 --gpu_ids 1  --results_dir results/hist_norm
```


## Train DSL on brain BraTS18, and Nuclei datasets(Distributed Version).

- Run the commands under `FedML` folder.

###Nuclei task

- train DSL on pathology data
```
nohup sh run_asdgan_distributed_pytorch.sh 5 n_exp_2.yml> ./log_exp_path.txt 2>&1 &
```
- save synthetic images using epoch 200 model; the images are saved in ./results/asdgan/test_200/path_resnet_9blocks_epoch200_experiment_0.h5
```
python save_syn.py --cfg n_exp_2.yml --batch_size 1 --save_dir ./run/path/asdgan/experiment_0 --epoch 200 --GPUid 0 --save_data
```

- save synthetic images for visualization (some random samples); the images are saved in ./results/asdgan/test_200/
```
python save_syn.py --cfg n_exp_2.yml --batch_size 1 --save_dir ./run/path/asdgan/experiment_0 --epoch 200 --GPUid 0
```


- run on background

```
nohup sh run_asdgan_distributed_pytorch.sh 4 n_exp_1.yml> ./log_nature_exp1.txt 2>&1 &
nohup sh run_asdgan_distributed_pytorch.sh 5 n_exp_2.yml> ./log_nature_exp2.txt 2>&1 &
nohup sh run_asdgan_distributed_pytorch.sh 4 n_exp_4.yml> ./log_nature_exp4.txt 2>&1 &
nohup sh run_asdgan_distributed_pytorch.sh 4 n_exp_4_miss_mod.yml> ./log_nature_exp4_mm.txt 2>&1 &
```

- save synthetic images

save 3 * 20 synthetic images to visualize

```
python save_syn.py --cfg n_exp_1.yml --batch_size 20 --save_dir ./run/heart/asdgan/experiment_0 --epoch 200 --GPUid 0 --num_test 3
python save_syn.py --cfg n_exp_2.yml --batch_size 20 --save_dir ./run/path/asdgan/experiment_0 --epoch 200 --GPUid 0 --num_test 3
python save_syn.py --cfg n_exp_4.yml --batch_size 20 --save_dir ./run/brats/asdgan_mc/experiment_0 --epoch 200 --GPUid 0 --num_test 3
```


save all synthetic images to h5 file for training segmentation model
```
python save_syn.py --cfg n_exp_1.yml --batch_size 1 --save_dir ./run/heart/asdgan/experiment_0 --epoch 200 --GPUid 0 --save_data
python save_syn.py --cfg n_exp_2.yml --batch_size 1 --save_dir ./run/path/asdgan/experiment_0 --epoch 200 --GPUid 0 --save_data
python save_syn.py --cfg n_exp_4.yml --batch_size 1 --save_dir ./run/brats/asdgan_mc/experiment_0 --epoch 200 --GPUid 0 --save_data
```

## calculate FID or Dist-FID

```
cd ./segmentation/utils/

python ./cal_fid.py [-h] --real_h5 REAL_H5 --fake_h5 FAKE_H5 [FAKE_H5 ...]
[--lv1_name LV1_NAME] [--oldformat] [--isrgb]
[--fake_h5_ch FAKE_H5_CH]

python ./cal_dist_fid.py [-h] --real_stats REAL_stats1 [REAL_stats2 ...] --fake_h5 FAKE_H5 [FAKE_H5 ...]
[--lv1_name LV1_NAME] [--oldformat] [--isrgb]
[--fake_h5_ch FAKE_H5_CH]

e.g. python cal_dist_fid.py --real_stats run/brats/asdgan_mc/experiment_0/client_0_data_stats.npz run/brats/asdgan_mc/experiment_0/client_1_data_stats.npz run/brats/asdgan_mc/experiment_0/client_2_data_stats.npz --fake_h5 datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch100_experiment_0.h5 datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch150_experiment_0.h5 datasets/asdgan_syn/brats_h5_all/brats_resnet_9blocks_epoch200_experiment_0.h5
```

