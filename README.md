# MRCNN
Pytorch implementation of MRCNN for SceneClassification. 

We refer to the structure in paper [Knowledge Guided Disambiguation for Large-Scale
Scene Classification with Multi-Resolution CNNs](https://arxiv.org/pdf/1610.01119.pdf)

## Datasets and Pretrained Models
 1. datasets URL: https://pan.baidu.com/s/1j4pSD0--Elo0QyAkmSOcBg
```Shell
ln -s $DOWNLOAD_PATH $MRCNN/data/
``` 
2. pretrained models for extra network URL: https://pan.baidu.com/s/11O7FQLyYC81WX4h7-1MPvA
```Shell
ln -s $DOWNLOAD_PATH $MRCNN/resnet/
```  
3.pretrained model for MRCNN URL: https://pan.baidu.com/s/10b8coHP3RW9yX-uAPmTauA      Extract code: 0hrr
```Shell
ln -s $DOWNLOAD_PATH $MRCNN/xq_model/
``` 
## Train 
cd $MRCNN/

run python train_MRCNN.py

## Test
cd $MRCNN/

mkdir xq_result

run python val_MRCNN.py

