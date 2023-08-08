

## PMMRec

This is the Torch implementation for our ICDE  2023 paper:

> Multi-Modality is All You Need for Transferable Recommender Systems

### Introduction

 In this paper, we unleash the boundaries of the ID-based paradigm and propose a Pure Multi-Modality based Recommender system (PMMRec), which relies solely on the multi-modal contents of the items (e.g., texts and images) and learns transition patterns general enough to transfer across domains and platforms. Specifically, we design a plug-and-play framework architecture consisting of multi- modal item encoders, a fusion module, and a user encoder. To align the cross-modal item representations, we propose a novel next-item enhanced cross-modal contrastive learning objective, which is equipped with both inter- and intra-modality negative samples and explicitly incorporates the transition patterns of user behaviors into the item encoders. To ensure the robustness of user representations, we propose a novel noised item detection objec- tive and a robustness-aware contrastive learning objective, which work together to denoise user sequences in a self-supervised manner. PMMRec is designed to be loosely coupled, so after being pre-trained on the source data, each component can be trans- ferred alone, or in conjunction with other components, allowing PMMRec to achieve versatility under both multi-modality and single-modality transfer learning settings. 



### Data Download and Processing

1. Dataset: 

   - We provide one processed **bili_food** dataset example . You first need to download [bili_food](https://drive.google.com/drive/folders/1A952wltpDkmnE-eCegyBrRT9BWa16fJQ?usp=drive_link), and put it in current file.

   - Get the lmdb of the images of the items

     ```shell
     python lmdb_build.py
     ```

     

2. Pre-trained PMMRec

   - We provide Pre-trained PMMRec . You fist need to download [Pmmrec_pt](https://drive.google.com/drive/folders/1c-pGz0UkG_Ki6V4YyjTsX5fOtNI2RW0a?usp=sharing), and put it in current file.

3. Download text encoders :  [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) , and put it in "**TextEncoders**"file.

4. Download text encoders :  [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) , and put it in "**CVEncoders**"file.



### Run

After processing the datasets, you can test the transfer learning of PMMrec  on Industrial dataset by:

```shell
cd ./src
python setup.py 
```

You also can set different parameters to train this model.