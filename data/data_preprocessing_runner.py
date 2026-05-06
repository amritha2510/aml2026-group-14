from pathlib import Path

from data.data_preprocessing import preprocess_dataset, save_preprocessed_metadata
from data.data_reader import (
    ChestXrayDataReader,
    get_preprocessing_grayscale,
    get_preprocessing_resize,
    get_required_config_path,
    load_config,
    load_metadata,
    save_metadata,
)


CONFIG_PATH = Path("config.yaml")
USE_SAVED_METADATA = True


def get_reader_and_config():
    config = load_config(CONFIG_PATH)
    reader = ChestXrayDataReader.from_yaml(CONFIG_PATH)
    return reader, config


def get_metadata(reader, config):
    metadata_output_path = get_required_config_path(config, CONFIG_PATH, "metadata_output_path")

    if USE_SAVED_METADATA:
        return load_metadata(metadata_output_path)

    metadata_df = reader.read_metadata()
    save_metadata(metadata_df, metadata_output_path)
    return metadata_df


def run_preprocessing():
    reader, config = get_reader_and_config()
    metadata_df = get_metadata(reader, config)

    if len(metadata_df) == 0:
        raise ValueError("No metadata rows found.")

    preprocessed_output_root = get_required_config_path(
        config, CONFIG_PATH, "preprocessed_output_root"
    )
    preprocessed_metadata_output_path = get_required_config_path(
        config, CONFIG_PATH, "preprocessed_metadata_output_path"
    )

    convert_to_grayscale = get_preprocessing_grayscale(config)
    resize_to = get_preprocessing_resize(config)

    preprocessed_df = preprocess_dataset(
        metadata_df=metadata_df,
        source_root=reader.data_root,
        target_root=preprocessed_output_root,
        convert_to_grayscale=convert_to_grayscale,
        resize_to=resize_to,
    )

    save_preprocessed_metadata(preprocessed_df, preprocessed_metadata_output_path)

    print(f"\n[INFO] Preprocessed images saved to: {preprocessed_output_root.resolve()}")
    print(f"[INFO] Preprocessed metadata saved to: {preprocessed_metadata_output_path.resolve()}")
    print(f"[INFO] Grayscale conversion: {'enabled' if convert_to_grayscale else 'disabled'}")

    if resize_to is None:
        print("[INFO] Resize disabled.")
    else:
        print(f"[INFO] Resize applied: {resize_to[0]}x{resize_to[1]}")


def main():
    run_preprocessing()


if __name__ == "__main__":
    main()