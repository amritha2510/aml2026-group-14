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
