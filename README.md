# PointTFA: Training-Free Clustering Adaption for Large 3D Point Cloud Models

# [Install environments]
The code is tested with CUDA==11.0 and pytorch==1.10.1\
```conda create -n tfa python=3.7.15``` \
```conda activate tfa``` \
```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge``` \
```pip install -r requirements.txt```

# [Download datasets and pre-trained models, put them in the right paths.]
Download the used datasets and pre-trained models from [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research). For now, you ONLY need to download "modelnet40_normal_resampled" and "ckpt_pointbert_ULIP-2.pt". \
After you download the datasets and pre-trained models. By default the data folder should have the following structure:
```
./DATA |
-- labels.json |
-- templates.json |
-- ./modelnet40
  -- modelnet40_normal_resampled |

./pretrained_ckpt |
-- ckpt_pointbert_ULIP-2.pt
```

# [Test on modelnet40] 
run ```main.py``` 
