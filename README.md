# aml2026-group-14

# Data 
1. data_reader.py  
   - Reusable data loading / metadata reading utilities  
   - Main reusable class: ChestXrayDataReader

2. data_analysis.py  
   - run by data_analysis_runner.py  
   - Includes:
     - dataset statistics
     - class distribution
     - image size / aspect ratio analysis
     - pixel intensity analysis
     - intensity outlier detection (for manual inspection)

3. data_preprocessing.py  
   - run by data_preprocessing_runner.py  
   - Resize configurable via config.yaml  
   - Produces preprocessed dataset + metadata


# Mental model
data_reader → reads raw dataset (no work needed atm)
preprocessing → creates new dataset (preprocess data before use)
models → use preprocessed dataset

# Current Data Pipeline
OFFLINE: Image → grayscale → resize (Configurable via config.yaml) → save
Runtime: → load → normalize ([0,1]) → model

# Evaluation

- Use metrics.py for a common evaluation  
- Primary metric: macro recall  
- Secondary: macro F1, confusion matrix, classification report  

Outputs:
- metrics.json  
- experiment_results.csv  
- run_summary.json  

Also saves:
- correct predictions (sample)  
- incorrect predictions (sample)  

# Logistic Regression Pipeline

Image → grayscale → resize → flatten → StandardScaler → (optional) PCA → Logistic Regression  

# LR - Model Selection

- Uses validation set only  
- Outputs:
  - model_selection_results.csv  
  - ranked results  
  - best config json  
  - plots  

Only best model evaluated on test.

# Open Points
- almost all images are grayscale, while some are RGB (but actually also grayscale), all moved to single channel.
- currently all resized to the same H*W (where H=W), independent of the aspect ratio, however the dataset has big difference in terms of their field of coverage and aspect ratio, therefore may require work there!
- Validation set has no viral, we have to change that and also confirm with the TA's - as it was rejected before (Maybe we need to explain we use the data is 3 classes, not 2)
- Confirm with TA's if we need to have the exact same preprocessing for all models, i.e. shoul we try to make each model perform its best, or compare them all using the exact same data input

## Check what visuals we would need to make a presentation at the end of the term, so that they can already be added