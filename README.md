# Attention-guided Temporal Coherent Video Object Matting

This is the Github project for our paper **Attention-guided Temporal Coherent Video Object Matting (arXiv:[2105.11427](https://arxiv.org/abs/2105.11427))**. We provide our code, the supplementary material, trained model and VideoMatting108 dataset here. For the trimap generation module, please see [TCVOM-TGM](https://github.com/yunkezhang/TCVOM-TGM).

**The code, the trained model and the dataset are for academic and non-commercial use only.**

The supplementary material can be found [here](https://1drv.ms/u/s!AuG441T6ysq5hWkSjlb29FSdM0Sc?e=LOCAHA).


## Table of Contents
  - [VideoMatting108 Dataset](#videomatting108-dataset)
  - [Models](#models)
  - [Usage](#usage)
    * [Requirements](#requirements)
    * [Inference](#inference)
    * [Training](#training)
  - [Contact](#contact)

## VideoMatting108 Dataset

VideoMatting108 is a large video matting dataset that contains 108 video clips with their corresponding groundtruth alpha matte, all in 1080p resolution, 80 clips for training and 28 clips for validation.

You can download the dataset [here](https://1drv.ms/u/s!AuG441T6ysq5g3pP5iTyOMRwUBDy?e=MdorID). The total size of the dataset is 192GB and we've split the archive into 1GB chunks.

The contents of the dataset are the following:

* ``FG``: contains the foreground RGBA image, where the alpha channel is the groundtruth matte and RGB channel is the groundtruth foreground.
* ``BG``: contains background RGB image used for composition.
* ``flow_png_val``: contains quantized optical flow of validation video clips for calculating ``MESSDdt`` metric. You can choose not to download this folder if you don't need to calculate this metric. You can refer to the `_flow_read()` function in `calc_metric.py` for usage.
* `*_videos*.txt`: train / val split.
* ``frame_corr.json``: FG / BG frame pair used for composition.

After decompressing, the dataset folder should have the structure of the following (please rename `flow_png_val` to `flow_png`):

```
|---dataset
  |-FG_done
  |-BG_done
  |-flow_png
  |-frame_corr.json
  |-train_videos.txt
  |-train_videos_subset.txt
  |-val_videos.txt
  |-val_videos_subset.txt
```

## Models

Currently our method supports four different image matting methods as base.

* `gca` (GCA Matting by Li et al., code is from [here](https://github.com/Yaoyi-Li/GCA-Matting))
* `dim` (DeepImageMatting by Xu et al., we use the reimplementation code from [here](https://github.com/poppinace/indexnet_matting))
* `index` (IndexNet Matting by Lu et al., code is from [here](https://github.com/poppinace/indexnet_matting))
* `fba` (FBA Matting by Forte et al., code is from [here](https://github.com/MarcoForte/FBA_Matting))
  * There are some differences in our training and the original FBA paper. We believe that there are still space for further performance gain through hyperparameter fine-tuning.
    * We did not use the foreground extension technique during training. Also we use four GPUs instead of one.
    * We used the conventional ``adam`` optimizer instead of ``radam``.
    * We used ``mean`` instead of ``sum`` during loss computation to keep the loss balanced (especially for `L_af`). 

The trained model can be downloaded [here](https://1drv.ms/u/s!AuG441T6ysq5hVLpbkAvbMaDNDmo?e=59i7f7). We provide four different weights for every base method.

* `*_SINGLE_Lim.pth`: The trained weight of the base image matting method on the VideoMatting108 dataset without TAM. Only `L_im` is used during the pretrain. **This is the baseline model.**
* `*_TAM_Lim_Ltc_Laf.pth`: The trained weight of base image matting method with TAM on VideoMatting108 dataset. `L_im`, `L_tc` and `L_af` is used during the training. **This is our full model.**
* `*_TAM_pretrain.pth`: The pretrained weight of base image matting method with TAM on the DIM dataset. Only `L_im` is used during the training.
* ``*_fe.pth``: The converted weight from the original model checkpoint, only used for pretraining TAM.

### Results

This is the quantitative result on VideoMatting108 validation dataset with `medium` width trimap. The metric is averaged on all 28 validation video clips.

**We use CUDA 10.2 during the inference. Using CUDA 11.1 might result in slightly lower metric. All metrics are calculated with ``calc_metric.py``.**

| Method             | Loss           | SSDA  | dtSSD | MESSDdt | MSE*(10^3) | mSAD  |
| ------------------ | -------------- | :---: | :---: | :-----: | :--------: | :---: |
| GCA+F (Baseline)   | L_im           | 55.82 | 31.64 |  2.15   |    8.20    | 40.85 |
| GCA+TAM            | L_im+L_tc+L_af | 50.41 | 27.28 |  1.48   |    7.07    | 37.65 |
| DIM+F (Baseline)   | L_im           | 61.85 | 34.55 |  2.82   |    9.99    | 44.38 |
| DIM+TAM            | L_im+L_tc+L_af | 58.94 | 29.89 |  2.06   |    9.02    | 43.28 |
| Index+F (Baseline) | L_im           | 58.53 | 33.03 |  2.33   |    9.37    | 43.53 |
| Index+TAM          | L_im+L_tc+L_af | 57.91 | 29.36 |  1.81   |    8.78    | 43.17 |
| FBA+F (Baseline)   | L_im           | 57.47 | 29.60 |  2.19   |    9.28    | 40.57 |
| FBA+TAM            | L_im+L_tc+L_af | 51.57 | 25.50 |  1.59   |    7.61    | 37.24 |

## Usage

### Requirements

```
Python=3.8
Pytorch=1.6.0
numpy
opencv-python
imgaug
tqdm
yacs
```

### Inference

`pred_single.py` and `pred_vmn.py` automatically use all CUDA devices available. `pred_test.py` uses `cuda:0` device as default.

* Inference on VideoMatting108 validation set using our full model

  * ```bash
    python pred_vmd.py --model {gca,dim,index,fba} --data /path/to/VideoMatting108dataset --load /path/to/weight.pth --trimap {wide,narrow,medium} --save /path/to/outdir
    ```

* Inference on VideoMatting108 validation set using the baseline model

  * ```bash
    python pred_single.py --dataset vmd --model {gca,dim,index,fba} --data /path/to/VideoMatting108dataset --load /path/to/weight.pth --trimap {wide,narrow,medium} --save /path/to/outdir
    ```

* Calculating metrics

  * ```bash
    python calc_metric.py --pred /path/to/prediction/result --data /path/to/VideoMatting108dataset
    ```

  * The result will be saved in `metric.json` inside `/path/to/prediction/result`. Use `tail` to see the final averaged result.

* Inference on test video clips

  * First, prepare the data. Make sure the workspace folder has the structure of the following:

    ```
    |---workspace
      |---video1
        |---00000_rgb.png
        |---00000_trimap.png
        |---00001_rgb.png
        |---00001_trimap.png
        |---....
      |---video2
      |---video3
      |---...
    ```

  * ```bash
    python pred_test.py --gpu CUDA_DEVICES_NUMBER_SPLIT_BY_COMMA --model {gca,vmn_gca,dim,vmn_dim,index,vmn_index,fba,vmn_fba} --data /path/to/workspace --load /path/to/weight.pth --save /path/to/outdir [video1] [video2] ...
    ```

    * The `model` parameter: `vmn_BASEMETHOD` corresponds to our full model, `BASEMETHOD` corresponds to the baseline model.
    * Without specifying the name of the video clip folders in the command line, the script will process all video clips under `/path/to/workspace`.

### Training

```bash
PY_CMD="python -m torch.distributed.launch --nproc_per_node=NUMBER_OF_CUDA_DEVICES"
```

* Pretrain TAM on DIM dataset. Please see `cfgs/pretrain_vmn_BASEMETHOD.yaml` for configuration and refer to `dataset/DIM.py` for dataset preparation.

  ```bash
  $PY_CMD pretrain_ddp.py --cfg cfgs/pretrain_vmn_index.yaml
  ```

* Training our full method on VideoMatting108 dataset. This will load the pretrained TAM weight as initialization. Please see `cfgs/vmd_vmn_BASEMETHOD_pretrained_30ep.yaml` for configuration.

  ```bash
  $PY_CMD train_ddp.py --cfg /path/to/config.yaml
  ```

* Training the baseline method on VideoMatting108 dataset without TAM. Please see `cfgs/vmd_vmn_BASEMETHOD_pretrained_30ep_single.yaml` for configuration.

  ```bash
  $PY_CMD train_single_ddp.py --cfg /path/to/config.yaml
  ```

## Contact

If you have any questions, please feel free to contact `yunkezhang@zju.edu.cn`.

