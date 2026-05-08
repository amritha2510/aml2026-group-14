from pathlib import Path

from data_analysis import (
    build_analysis_table,
    compute_image_intensity_stats,
    get_intensity_outliers,
    plot_aspect_ratio_distribution,
    plot_class_distribution,
    plot_image_mode_distribution,
    plot_image_size_distributions,
    plot_intensity_by_class,
    plot_intensity_stats_distribution,
    plot_mean_vs_std,
    plot_pixel_intensity_distribution,
    print_aspect_ratio_summary,
    print_basic_summary,
    print_class_proportions,
    print_image_mode_summary,
    print_intensity_stats_summary,
    print_resize_suggestion,
    save_class_weights,
    save_summary_tables,
    show_sample_images,
    show_sample_rgb_files,
)
from data_reader import (
    ChestXrayDataReader,
    get_required_config_path,
    load_config,
    load_metadata,
    save_metadata,
)


CONFIG_PATH = Path("/home/renku/work/aml2026-group-14/config.yaml")
USE_SAVED_METADATA = False


def get_reader_and_config():
    config = load_config(CONFIG_PATH)
    reader = ChestXrayDataReader.from_yaml(CONFIG_PATH)
    return reader, config


def get_metadata(reader, config):
    metadata_output_path = get_required_config_path(
        config,
        CONFIG_PATH,
        "metadata_output_path",
    )

    if USE_SAVED_METADATA:
        return load_metadata(metadata_output_path)

    metadata_df = reader.read_metadata()
    save_metadata(metadata_df, metadata_output_path)
    return metadata_df


def run_analysis():
    reader, config = get_reader_and_config()
    metadata_df = get_metadata(reader, config)

    if len(metadata_df) == 0:
        raise ValueError("No metadata rows found.")

    analysis_output_dir = get_required_config_path(
        config,
        CONFIG_PATH,
        "analysis_output_dir",
    )

    analysis_df = build_analysis_table(metadata_df)

    print_basic_summary(analysis_df)
    print_image_mode_summary(analysis_df)
    print_resize_suggestion(analysis_df)
    print_class_proportions(analysis_df)
    print_aspect_ratio_summary(analysis_df)

    show_sample_rgb_files(analysis_df, n=10)

    save_summary_tables(analysis_df, analysis_output_dir)
    plot_class_distribution(analysis_df, analysis_output_dir)
    plot_image_mode_distribution(analysis_df, analysis_output_dir)
    plot_image_size_distributions(analysis_df, analysis_output_dir)
    plot_aspect_ratio_distribution(analysis_df, analysis_output_dir)
    plot_pixel_intensity_distribution(analysis_df, analysis_output_dir)
    show_sample_images(analysis_df, analysis_output_dir, split="train")

    intensity_stats_df = compute_image_intensity_stats(analysis_df)

    if len(intensity_stats_df) > 0:
        intensity_stats_df.to_csv(
            Path(analysis_output_dir) / "image_intensity_stats.csv",
            index=False,
        )
        print_intensity_stats_summary(intensity_stats_df)
        plot_intensity_stats_distribution(intensity_stats_df, analysis_output_dir)
        plot_mean_vs_std(intensity_stats_df, analysis_output_dir)
        plot_intensity_by_class(analysis_df, analysis_output_dir)
        outliers_df = get_intensity_outliers(intensity_stats_df, 10)

        print("\nIntensity outliers selected for manual review:")
        print(outliers_df[["filepath", "outlier_reason", "mean_intensity", "std_intensity", "min_intensity", "max_intensity"]].to_string(index=False))
    else:
        print("[WARNING] No intensity statistics were computed.")

    class_weights = save_class_weights(analysis_df, analysis_output_dir)
    print("\nSuggested class weights from train split:")
    print(class_weights)

    print(f"\n[INFO] Analysis outputs saved to: {Path(analysis_output_dir).resolve()}")


def main():
    run_analysis()


if __name__ == "__main__":
    main()