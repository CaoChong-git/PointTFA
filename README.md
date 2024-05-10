# PointTFA: Training-Free Clustering Adaption for Large 3D Point Cloud Models

# Introduction
PointTFA is a training-free adaptation of large 3D point cloud models. This approach effectively bridges the domain gap between ULIP pre-training and downstream point clouds. We propose a "Data-efficiency" (RMC) to capture robust representative knowledge in support memory, forming an efficient cache model that balances data dimension and baseline accuracy. Additionally, we present a reshaping technique (CQR) for projecting cache knowledge onto test features, enhancing the comprehensive understanding of cache information by test point clouds. Furthermore, PointTFA significantly improves the original capabilities of the ULIP pre-trained 3D backbone, achieving state-of-the-art performance in Training-free few-shot 3D Classification.

<div align="center">
  <img width=900 src="framework.png"/>
</div>

## Install environments
The code is tested with CUDA==11.0 and pytorch==1.10.1\
```conda create -n tfa python=3.7.15``` \
```conda activate tfa``` \
```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge``` \
```pip install -r requirements.txt```

## Download datasets and pre-trained models, put them in the right paths
Download the used datasets and pre-trained models from [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research). For now, you ONLY need to download "modelnet40_normal_resampled" and "ckpt_pointbert_ULIP-2.pt". \
After you download the datasets and pre-trained models. By default the data folder should have the following structure:
```
./DATA |
-- labels.json |
-- templates.json |
-- /modelnet40
  -- modelnet40_normal_resampled |

./pretrained_ckpt |
-- ckpt_pointbert_ULIP-2.pt
```

## Get Started
### Configs
The running configurations can be modified in `configs/dataset.yaml`. 

For simplicity, we provide the hyperparamters achieving the overall best performance on 1\~16 shots for a dataset, which accord with the scores reported in the paper. If respectively tuned for different shot numbers, the 1\~16-shot performance can be further improved. You can edit the `search_scale`, `search_step`, `init_beta` and `init_alpha` for fine-grained tuning.

Note that the default `load_cache` and `load_pre_feat` are `False` for the first running, which will store the cache model and test features in `cache/dataset/`. For later running, they can be set as `True` for faster hyperparamters tuning.

### Running
For ModelNet40 dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/modelnet40.yaml
