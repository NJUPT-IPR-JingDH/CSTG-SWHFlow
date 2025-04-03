# CSTG-SWHFlow
We discover that many low-light RGB images contain lots of color pixels with zeros caused by the low-light, which means that the low-light images suffer low-light, noise, and information loss. In this paper, we propose a novel normalizing flow learning based method CSTG-SWHFlow for low-light enhancement, noise suppression and lost information restoration of zero-element pixels, which consists of a compressive self-attention transformer based conditional generator (CSTG) and a spatial/width/height-axis fusion attention network driven flow (SWHFlow). On the one hand, height-and spatial-axis self-attention transformer (HST) is designed to successively cascade height-and spatial-axis self-attention (HSA) and multi-scale fusion feed forward network (MF-FFN). Based on HST, height-and spatial-axis self-attention with zero-map transformer (HSZT) is designed to perform weight fusion between height/spatial-axis self-attention and zero-map height/spatial-axis self-attention. CSTG is designed as a U-shape network developed by HSZT and HST. On the other hand, we design a spatial/width/height-axis fusion attention network (SWHF) to learn efficient scale and shift parameters for affine coupling/injector layers of the normalizing flow, to construct our SWHFlow, where a multi-scale fusion spatial attention module (MFS) is designed to encode the conditional feature. Experiments show that CSTG-SWHFlow outperforms existing SOTA methods on benchmark low-light datasets, and low-light images with massive zero-element pixels.

## Dataset
LOLv2 (Real & Synthetic): Please refer to the papaer [From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (CVPR 2020)](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)

SID & SMID & SDSD (indoor & outdoor): Please refer to the paper [SNR-aware Low-Light Image Enhancement (CVPR 2022)](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance)

## Testing

### Pre-trained Models
The pretrained models can be downloaded [HERE](https://pan.baidu.com/s/1HsZeKHTo7i7ZBRou_XKMBg?pwd=4jxa) .


## Run the testing code

You need to specify the model path `model_path` in the config file.  
Then run:

```bash
python test_for_lol.py
```


Dataroots of both training and testing can be changed on `datasets/val/root` of config file. Dataroot of training can be changed on `datasets/train/root`

## Trainning
To train on your own datasets, you need to specify the path of the dataroot in config file, then run:

```bash
python train.py
```

the root of dataset should be like this:
```bash
Your_own_datasets
      |_train
        |__low
        |__high
      |_test
        |__low
        |__high
```
