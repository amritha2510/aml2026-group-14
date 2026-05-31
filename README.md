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

# Evaluation

- Use metrics.py for a common evaluation  
- Primary metric: macro recall  
- Secondary: macro F1, confusion matrix, classification report  

Outputs:
- metrics.json  
- experiment_results.csv  
- run_summary.json  
