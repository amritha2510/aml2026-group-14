## Project Idea: Pneumonia Classification 

We want to use the [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle to classify chest X-ray images into **normal, bacterial pneumonia, and viral pneumonia**, and see how well Vision Transformers work on this task.

The dataset has **~5,800 images** split into train/validation/test. It’s relatively small and **imbalanced**, especially in the validation and test sets, which we’ll take into account.

### Plan  
- Treat it as a **3-class classification problem** (normal / bacterial / viral)  (Normal means clean lungs, bacterial means lobular, viral means spread clouds in lungs)
- Fine-tune a pretrained **Vision Transformer**
- Evaluation setup: **TBD**  
- Baselines / comparisons: **TBD**  
- Potentially adjust the provided train/val/test split due to imbalance
- Handle imbalance via experimentation over data augmentation or weighting   


# Problem Setting: Chest X-Ray Pneumonia Classification

## 1. Formal Problem Setting
Our objective is to classify chest X-ray images into three distinct categories: Normal (clean lungs), Bacterial Pneumonia (lobular), and Viral Pneumonia (spread clouds). 

Formally, given a dataset of chest X-ray images $D = \{(x_i, y_i)\}_{i=1}^N$, let $x_i \in \mathbb{R}^{H \times W \times C}$ represent the $i^{th}$ input image, where $H$, $W$, and $C$ denote the height, width, and number of channels, respectively. Our goal is to predict the corresponding label $y_i \in \{0, 1, 2\}$, where:
* $y = 0$: Normal
* $y = 1$: Bacterial Pneumonia
* $y = 2$: Viral Pneumonia

We aim to learn a parameterized mapping function $f_{\theta}: x \rightarrow y$ that maximizes the classification performance on unseen data. 

## 2. Evaluation Protocol
We will utilize the Chest X-Ray Pneumonia dataset from Kaggle, which contains approximately 5,800 images. Because the provided train, validation, and test sets are heavily imbalanced, we will adjust the splits using stratified sampling to ensure a representative distribution of all three classes across our evaluation splits. 

**Metrics:** Because this is an imbalanced multiclass classification problem, standard accuracy is insufficient. Our problem definition leads to **Multiclass AUROC** (Area Under the Receiver Operating Characteristic curve) and **Macro F1-Score** as the natural best choices for our primary evaluation metrics. 

Note: We will strictly use this exact same evaluation protocol and data split to evaluate all our baseline models and our final proposed model.

## 3. Baselines
To evaluate our model in a meaningful context, we will compare it against two baselines:

* **Simple/Statistical Baseline:** A Random Forest classifier trained on flattened, downsampled versions of the images (or utilizing basic PCA feature extraction). This satisfies the requirement to include a "simple" machine learning approach.
* **Machine Learning Baseline:** A standard Convolutional Neural Network (e.g., ResNet-50 or DenseNet-121) trained on the same dataset. This serves as a strong traditional benchmark for medical image classification.

## 4. Proposed Model & Hyperparameter Tuning


Our primary model will be a **Vision Transformer (ViT)**, which satisfies the requirement to use an architecture covered after LSTMs. 

Because ViTs lack the spatial inductive biases of CNNs and are highly data-hungry, training from scratch on 5,800 images would likely lead to severe overfitting. Therefore, we will utilize a pre-trained ViT model (e.g., pre-trained on ImageNet) and fine-tune it on our specific dataset.

To address the class imbalance, we will experiment with data augmentation techniques and class weighting during training. We will also tune the hyperparameters of the ViT in a reasonable scope, specifically focusing on the learning rate, weight decay, and dropout rates..