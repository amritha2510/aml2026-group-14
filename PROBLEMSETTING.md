## Project Idea: Pneumonia Classification 

We want to use the [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle to classify chest X-ray images into **normal, bacterial pneumonia, and viral pneumonia**, and see how well Vision Transformers work on this task.

The dataset has **~5,800 images** split into train/validation/test. It’s relatively small and **imbalanced**, especially in the validation and test sets, which we’ll take into account.

# Problem Setting: Chest X-Ray Pneumonia Classification

## 1. Formal Problem Setting
Our objective is to classify chest X-ray images into three distinct categories: Normal (clean lungs), Bacterial Pneumonia (lobular), and Viral Pneumonia (spread clouds). 

Formally, given a dataset of chest X-ray images $D = \{(x_i, y_i)\}_{i=1}^N$, let $x_i \in \mathbb{R}^{H \times W \times C}$ represent the $i^{th}$ input image, where $H$, $W$, and $C$ denote the height, width, and number of channels, respectively. Our goal is to predict the corresponding label $y_i \in \{0, 1, 2\}$, where:
* $y = 0$: Normal
* $y = 1$: Bacterial Pneumonia
* $y = 2$: Viral Pneumonia

We aim to learn a parameterized mapping function $f_{\theta}: x \rightarrow y$ that minimizes our chosen cost function on unseen data. 

## 2. Evaluation Protocol
We will utilize the Chest X-Ray Pneumonia dataset from Kaggle, which contains approximately 5,800 images. Because the provided train, validation, and test sets are heavily imbalanced, we will adjust the splits using stratified sampling to ensure a representative distribution of all three classes across our evaluation splits. 

**Primary Metric:** In a medical diagnostic context, missing a pneumonia diagnosis (a False Negative) is the worst-case scenario that must be avoided. Therefore, our primary evaluation metric will be **Macro Recall (Sensitivity)**, aiming to minimize missed diagnoses across all classes. We will also track the Macro F1-Score for broader context.

**Note**: We will use the exact same evaluation protocol and data split to evaluate all our baseline models and our final proposed model.

## 3. Baselines
To evaluate our model in a meaningful context, we will compare it against three baselines:

1. **Simple/Statistical Baseline:** A Logistic Regression classifier trained on flattened, downsampled versions of the images (utilizing PCA for dimensionality reduction and feature extraction). This satisfies the requirement to include a "simple" statistical approach to compare our more complex models against.
2. **Machine Learning Baseline:** A standard Convolutional Neural Network (e.g., ResNet-50 or DenseNet-121) trained on the same dataset. This serves as a strong traditional benchmark for medical image classification.
3. **Standard ViT Baseline:** A standard, pre-trained Vision Transformer simply fine-tuned on our dataset, serving as the direct baseline for our custom architectural changes.

## 4. Proposed Model & Hyperparameter Tuning
Our primary proposed model builds upon the Vision Transformer (ViT), which satisfies the requirement to use an architecture covered after LSTMs. Because standard ViTs lack the spatial inductive biases of CNNs and are notoriously data-hungry, simply fine-tuning from scratch on 5,800 images leads to overfitting and brittle representations.

**Our Architectural Twist (Dual-Branch Feature Fusion):** 
Instead of a standard sequential model, we will use a parallel dual-branch architecture:
* **Stream 1 (Local Focus):** A lightweight CNN branch (e.g., early layers of ResNet-18) to extract fine-grained, lower-level textural features of the pneumonia opacities.
* **Stream 2 (Global Focus):** A pre-trained ViT branch to process the original image and extract higher-level, global structural context of the chest cavity. 

**Noise Injection for Generalization:**
The final feature representations from both branches will be concatenated (feature fusion) before the final classification layer. Because ViTs are highly prone to memorizing small datasets, we will introduce **embedding-level noise injection** (via targeted heavy Dropout) specifically at the end of the ViT branch before the fusion. This acts as a strict regularizer, forcing the model to rely on both local CNN textures and global ViT structure, thereby reducing memorization and improving generalization.

To further address the class imbalance, we will experiment with data augmentation techniques and class weighting during training. 
We will also tune the hyperparameters of this custom model in a reasonable scope, specifically focusing on the learning rate, the noise/dropout rate, and weight decay.


