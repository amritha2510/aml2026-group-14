import pandas as pd
from sklearn.model_selection import train_test_split


def fix_val_split(metadata_df, val_fraction=0.15, random_state=42):
    """
    Re-splits train into train+val with stratification.
    Keeps the original test set untouched.
    
    Use this if the original val set is missing a class (viral).
    """
    train_df = metadata_df[metadata_df["split"] == "train"].copy()
    test_df = metadata_df[metadata_df["split"] == "test"].copy()
    # Drop original val entirely
    
    train_new, val_new = train_test_split(
        train_df,
        test_size=val_fraction,
        stratify=train_df["label"],
        random_state=random_state,
    )
    train_new = train_new.copy()
    val_new = val_new.copy()
    train_new["split"] = "train"
    val_new["split"] = "val"

    result = pd.concat([train_new, val_new, test_df], ignore_index=True)
    
    print("[split_utils] New split distribution:")
    print(result.groupby(["split", "label"]).size())
    
    return result