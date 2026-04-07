# aml2026-group-14

1. data_reader.py
   - Reusable data loading / metadata reading utilities
   - Main reusable class: ChestXrayDataReader

2. data_analysis.py
   - Summaries, plots, CSV outputs, class weights

3. data_preprocessing.py

4. runner.py // separated now..
   - Example pipeline using the separated modules

# Mental model
data_reader → reads raw dataset
preprocessing → creates new dataset
models → use preprocessed dataset

# Current Data Pipeline
Image → grayscale → resize (Configurable via config.yaml)

# Logistic Regression Pipeline

- Image → grayscale → resize (64x64) → flatten → PCA → Logistic Regression