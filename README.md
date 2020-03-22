## Bengali.AI Handwritten Grapheme Classification 2019-2020

Our team ranked 254th place (TOP 13%) in the 2-stage competition [Bengali.AI Handwritten Grapheme Classification 2019-2020 competition on Kaggle platform](https://www.kaggle.com/c/bengaliai-cv19/leaderboard) with inference in kernels. This repository consists of code and utils that I used to train our models. The solution is powered by awesome [Albumentations](https://github.com/albu/albumentations), [PyTorch](https://pytorch.org), [Cadene models](https://github.com/Cadene/pretrained-models.pytorch), [EfficientNet](https://github.com/rwightman/pytorch-image-models) and [FastAi](https://docs.fast.ai/) libraries.

## In this repository you can find:
* `se_resnext50_32x4d_train_by_folds` - folder with full simple pipeline:
  * how I prepared folds
  * what utils did I use
  * how I trained and finetuned the model **(SE_ResNext50_32x4d)**
* `models`:
  * `se_resnext50_32x4d_0fold_example.ipynb` - Best single model by CV. Example with training first fold
  * `densenet121_128x128.ipynb` - Best single model by LB. We could take 72nd place (TOP 4%) with this model, but we didn't choose this as the final one.
  * `effnetb3_imgsize128_1channel-fp32.ipynb` - second single model by CV
* `preprocessing_to_npy.ipynb` - script with how to save `.parquet` images into `.npy` (and resized it)

## Solution description:

### Data
The task proposed in this competition is recognition of handwritten Bengali letters. In contrast to similar competitions such as [mnist digit recognition](https://www.kaggle.com/c/digit-recognizer) or the recent [Kannada MNIST](https://www.kaggle.com/c/Kannada-MNIST), Bengali alphabet is quite complex and may include ~13,000 different grapheme variations. Fortunately, each grapheme can be decomposed into 3 parts: grapheme_root, vowel_diacritic, and consonant_diacritic (168, 11, and 7 independent classes, respectively). Therefore, the task of grapheme recognition is significantly simplified in comparison with 13k-way classification. Though, additional consideration may be required for this multitask classification, like checking if 3 independent models, or a single model one head, or a single model with 3 heads (our approach) works the best.

### Models and huge shake-up
We tried many preprocessing and augmentations strategies in this competition, but we achieved our two best single results by two models: `se_resnext50_32x4d` and `effnetb3` with cutmix/mixup approach and some augmentations. ***These were the best models on CV, but in the private test part there were a lot of new grapheme combinations that we were not ready for. This was the main reason for the huge shake-up for most of the participants.*** There's a [great illustration](https://www.kaggle.com/c/bengaliai-cv19/discussion/136054) of what we've all been watching.

According to the rules, in the Stage 1 submission we uploaded all our models. And then these models will be used to generate the final submissions for scoring in Stage 2. Public leaderboard in Stage 1 is calculated with approximately ~46% of the test data.

For preprocessing we use invert and resize images with (or without) cropping keeping only the characters.
Good kernel with a similar approach right [here](https://www.kaggle.com/iafoss/image-preprocessing-128x128).

### Augmentations
* Cutmix / Mixup - showed excellent metric growth.
* From [Albumentations](https://github.com/albu/albumentations) library: ShiftScaleRotate, IAAPerspective and IAAPiecewiseAffine worked best for me.

### Training
All models in this repository were trained using FastAi and Pytorch libraries. Adam as optimizer with ReduceLROnPlateau or OneCycle policy gave me the best results.

### Hardware
We used 2x* *2080*, 8x* *1080* and GCP credits.

## Team:
- Mishunyayev Nikita: [Kaggle](https://www.kaggle.com/mnikita), [GitHub](https://github.com/Mishunyayev-Nikita)
- Andrew Lukyanenko: [Kaggle](https://www.kaggle.com/artgor), [GitHub](https://github.com/Erlemar)
- Vlad A: [Kaggle](https://www.kaggle.com/valyukov), [GitHub](https://github.com/valyukov)
- Ilya Dobrynin: [Kaggle](https://www.kaggle.com/ilyadobrynin)
- Ivan Panshin: [Kaggle](https://www.kaggle.com/ivanpan)

## How the winner's decision was different from ours:
[**1st place**](https://www.kaggle.com/c/bengaliai-cv19/discussion/135984):
 * All models classify against the 14784 (168 * 11 * 8) classes
 * No cutmix/mixup, just cutout
 * Different training stages:
   * Effnet-B7 for classify to "Seen" or "Unseen" images;
   * Effnet-B7 for classify 1295 classes included in the training data;
   * Classifier for images synthesized from ttf files and generator that converts handwritten characters into the desired synthesized data-like image
   
[**2nd place**](https://www.kaggle.com/c/bengaliai-cv19/discussion/135966):
* Switch from predicting R,C,V to predicting individual graphemes
* [Fmix](https://arxiv.org/abs/2002.12047) worked clearly better than cutmix, and also the resulting images looked way more natural to us due to the way the cut areas are picked
* Postprocessing (read the post)

[**3rd place**](https://www.kaggle.com/c/bengaliai-cv19/discussion/135982):
* Replace softmax with [pc-softmax](https://arxiv.org/abs/1911.10688) and use negative log likelihood as loss function
* Сlassifier to "Seen" or "Unseen" images
* Arcface approach:
  * use cosine similarity between train and test embedding feature;
  * threshold: smallest cosine similarity between train and validation embedding feature

[**5th place**](https://www.kaggle.com/c/bengaliai-cv19/discussion/136129):
* 4 cycle cosine annealing with augmentation increase
* Recoded the cons classes into a multilabel classification problem (read the post)
* Losses:
  * root/vowel loss: CrossEntropy
  * consonant loss: Multi label Binary Crossentropy
  * grapheme loss: ArcCos + CrossEntropy
  * total loss = root loss + consonant loss + vowel loss + grapheme loss
 
[**7th place**](https://www.kaggle.com/c/bengaliai-cv19/discussion/135960):
* Finetuned previous models with the synthetic data (graphemes are actually encoded as sequence of unicode characters)

[**10th place**](https://www.kaggle.com/c/bengaliai-cv19/discussion/136815):
* Single model in 4 days
* Instead of simply applying Global Average Pooling to feature map and using its result as common input of each component's head, He used [sSE Block](https://arxiv.org/abs/1803.02579) and Global Average Pooling for each component. Examples: [post+kernel](https://www.kaggle.com/c/bengaliai-cv19/discussion/137552) and [comment here](https://www.kaggle.com/c/bengaliai-cv19/discussion/136815#781819)
* No cutmix / mixup, just hard augmentations with RandomErasing

[**12nd place**](https://www.kaggle.com/c/bengaliai-cv19/discussion/135998):
* It's a simple enough solution with blending different architectures, augmentations and parameters
* The approach is similar to ours, but a more accurate selection of parameters allowed them not to ovetfit as us, good job!

[**13th**](https://www.kaggle.com/c/bengaliai-cv19/discussion/136116) and [**14th**](https://www.kaggle.com/c/bengaliai-cv19/discussion/136021) places:
* Hacking Macro Recall as postprocessing
* CAM CutMix (2-stage training)

Other Top Solutions you can find [here](https://www.kaggle.com/c/bengaliai-cv19/discussion/136769).
