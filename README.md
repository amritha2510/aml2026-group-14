# aml2026-group-14

# Data 
1. data_reader.py
   - Reusable data loading / metadata reading utilities
   - Main reusable class: ChestXrayDataReader

2. data_analysis.py
   - run by data_analysis_runner.py

3. data_preprocessing.py
   - run by data_preprocessing_runner.py (Resize configurable via config.yaml)

# Mental model
data_reader → reads raw dataset (no work needed atm)
preprocessing → creates new dataset (preprocess data before use)
models → use preprocessed dataset

# Current Data Pipeline
Image → grayscale → resize (Configurable via config.yaml)

# Evaluation
- Use metrics.py for a common evaluation
- Use "macro recall" as primary, and take the rest as secondary (i.e. diagnostics)
- Add more if you have additions, so that it will be easy to compare the models and prepare the presentation

# Logistic Regression Pipeline
- Image → grayscale → resize (64x64) → flatten → PCA → Logistic Regression

# Open Points
- almost all images are grayscale, while some are RGB (but actually also grayscale), all moved to single channel.
- currently all resized to the same H*W (where H=W), independent of the aspect ratio, however the dataset has big difference in terms of their field of coverage and aspect ratio, therefore may require work there!
- Validation set has no viral, we have to change that and also confirm with the TA's - as it was rejected before (Maybe we need to explain we use the data is 3 classes, not 2)
- Confirm with TA's if we need to have the exact same preprocessing for all models, i.e. shoul we try to make each model perform its best, or compare them all using the exact same data input


## TODO: validation should be a valid set, and should be the only one used in fine-tuning
## Check what visuals we would need to make a presentation at the end of the term, so that they can already be added