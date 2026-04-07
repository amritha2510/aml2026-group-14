from pathlib import Path

from data.data_analysis import (
    compute_class_weights,
    plot_class_distribution,
    plot_image_size_distributions,
    plot_pixel_intensity_distribution,
    print_basic_summary,
    print_resize_suggestion,
    save_class_weights,
    save_summary_tables,
    show_sample_images,
)
from data.data_preprocessing import enrich_metadata_with_image_stats
from data.data_reader import ChestXrayDataReader


DATA_ROOT = Path("./chest_xray")
SAVE_DIR = Path("outputs/data_analysis")


def main() -> None:
    reader = ChestXrayDataReader(DATA_ROOT)

    print("[INFO] Building metadata table...")
    df = reader.read_metadata()

    if len(df) == 0:
        raise ValueError(
            f"No images found under {DATA_ROOT}. "
            "Check your DATA_ROOT path and dataset structure."
        )

    print("[INFO] Inspecting images...")
    df = enrich_metadata_with_image_stats(df)

    print_basic_summary(df)
    print_resize_suggestion(df)

    class_weights = compute_class_weights(df)
    print("\nSuggested class weights from train split:")
    print(class_weights)

    save_summary_tables(df, SAVE_DIR)
    plot_class_distribution(df, SAVE_DIR)
    plot_image_size_distributions(df, SAVE_DIR)
    plot_pixel_intensity_distribution(df, SAVE_DIR)
    show_sample_images(df, SAVE_DIR, split="train")
    save_class_weights(df, SAVE_DIR)

    print(f"\n[INFO] Data analysis completed. Outputs saved to: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
