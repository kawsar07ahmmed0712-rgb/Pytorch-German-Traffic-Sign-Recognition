# src/helper_functions.py

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch


def set_seeds(seed: int = 42) -> torch.Generator:
    """
    Set random seeds for reproducibility.

    Returns:
        torch.Generator: A seeded generator that can be used in DataLoader.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def make_project_dirs(*paths: Path) -> None:
    """
    Create project directories if they do not already exist.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_gtsrb_csvs(data_dir: Path):
    """
    Load GTSRB train, test, and meta CSV files.

    Args:
        data_dir: Path to the Data folder.

    Returns:
        train_df, test_df, meta_df
    """
    train_df = pd.read_csv(data_dir / "Train.csv")
    test_df = pd.read_csv(data_dir / "Test.csv")
    meta_df = pd.read_csv(data_dir / "Meta.csv")

    return train_df, test_df, meta_df


def add_full_image_path(df: pd.DataFrame, data_dir: Path, path_col: str = "Path") -> pd.DataFrame:
    """
    Add a full image path column using the dataset path column.

    Args:
        df: Input dataframe.
        data_dir: Path to the Data folder.
        path_col: Column containing relative image paths.

    Returns:
        DataFrame with a new full_path column.
    """
    df = df.copy()
    df["full_path"] = df[path_col].apply(lambda x: str(data_dir / x))
    return df


def extract_track_id(path: str) -> str:
    """
    Extract a possible track/group ID from a GTSRB image path.

    Example:
        Train/20/00020_00000_00004.png
        track_id -> 00020_00000

    This helps us later check whether grouped validation splitting is possible.
    """
    stem = Path(path).stem
    parts = stem.split("_")

    if len(parts) >= 2:
        return "_".join(parts[:2])

    return stem


def check_path_exists(df: pd.DataFrame, path_col: str = "full_path", n: int = 10) -> pd.DataFrame:
    """
    Check whether sample image paths exist.

    Args:
        df: Input dataframe.
        path_col: Column containing full image paths.
        n: Number of rows to check.

    Returns:
        Small dataframe with path existence result.
    """
    sample_df = df[[path_col]].head(n).copy()
    sample_df["exists"] = sample_df[path_col].apply(lambda x: Path(x).exists())
    return sample_df




### Plotting funcitons


import matplotlib.pyplot as plt


def get_class_distribution(df: pd.DataFrame, label_col: str = "ClassId") -> pd.DataFrame:
    """
    Create a class distribution table.

    Args:
        df: Input dataframe.
        label_col: Column containing class labels.

    Returns:
        DataFrame with class id and image count.
    """
    class_counts = (
        df[label_col]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    class_counts.columns = [label_col, "image_count"]
    return class_counts


def plot_class_distribution(
    class_counts: pd.DataFrame,
    label_col: str = "ClassId",
    count_col: str = "image_count",
    title: str = "Class Distribution",
    figsize: tuple = (14, 5)
):
    """
    Plot class distribution as a bar chart.

    Args:
        class_counts: DataFrame containing class labels and counts.
        label_col: Class label column name.
        count_col: Count column name.
        title: Plot title.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    plt.bar(class_counts[label_col].astype(str), class_counts[count_col])
    plt.title(title)
    plt.xlabel("Class ID")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def clean_sheet_name(name: str) -> str:
    """
    Clean Excel sheet name.

    Excel sheet names cannot be longer than 31 characters and cannot contain
    some special characters.
    """
    invalid_chars = ["\\", "/", "*", "?", ":", "[", "]"]

    for char in invalid_chars:
        name = name.replace(char, "_")

    return name[:31]


def save_tables_to_excel(tables: dict, file_path: Path) -> None:
    """
    Save multiple small pandas DataFrames into one Excel workbook.

    Args:
        tables: Dictionary where keys are sheet names and values are DataFrames.
        file_path: Output Excel file path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            clean_name = clean_sheet_name(sheet_name)

            if isinstance(table, pd.Series):
                table = table.reset_index()

            table.to_excel(writer, sheet_name=clean_name, index=False)

    print(f"Saved {len(tables)} tables to: {file_path}")


def save_or_update_tables_to_excel(tables: dict, file_path: Path) -> None:
    """
    Save or update multiple small DataFrames into one Excel workbook.

    If the workbook already exists, only matching sheets are replaced.
    Other existing sheets are kept.

    Args:
        tables: Dictionary where keys are sheet names and values are DataFrames.
        file_path: Output Excel file path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    writer_mode = "a" if file_path.exists() else "w"

    if writer_mode == "a":
        writer = pd.ExcelWriter(
            file_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace"
        )
    else:
        writer = pd.ExcelWriter(
            file_path,
            engine="openpyxl",
            mode="w"
        )

    with writer:
        for sheet_name, table in tables.items():
            clean_name = clean_sheet_name(sheet_name)

            if isinstance(table, pd.Series):
                table = table.reset_index()

            table.to_excel(writer, sheet_name=clean_name, index=False)

    print(f"Saved or updated {len(tables)} tables to: {file_path}")






def get_image_size_summary(df: pd.DataFrame) -> dict:
    """
    Create summary tables for image width and height.

    Args:
        df: DataFrame containing Width and Height columns.

    Returns:
        Dictionary of small DataFrames.
    """
    size_summary = df[["Width", "Height"]].describe().reset_index()

    unique_sizes = (
        df.groupby(["Width", "Height"])
        .size()
        .reset_index(name="image_count")
        .sort_values("image_count", ascending=False)
    )

    return {
        "size_summary": size_summary,
        "most_common_sizes": unique_sizes.head(20),
        "unique_size_count": pd.DataFrame({
            "metric": ["unique_widths", "unique_heights", "unique_width_height_pairs"],
            "value": [
                df["Width"].nunique(),
                df["Height"].nunique(),
                unique_sizes.shape[0],
            ],
        }),
    }


def plot_image_size_distribution(
    df: pd.DataFrame,
    title_prefix: str = "Image Size",
    figsize: tuple = (12, 4)
):
    """
    Plot width and height distributions.

    Args:
        df: DataFrame containing Width and Height columns.
        title_prefix: Prefix for plot title.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    plt.hist(df["Width"], bins=30, alpha=0.7, label="Width")
    plt.hist(df["Height"], bins=30, alpha=0.7, label="Height")
    plt.title(f"{title_prefix} Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Number of Images")
    plt.legend()
    plt.tight_layout()
    plt.show()



from PIL import Image


def show_random_images(
    df: pd.DataFrame,
    image_col: str = "full_path",
    label_col: str = "ClassId",
    n: int = 16,
    seed: int = 42,
    cols: int = 4,
    figsize: tuple = (10, 10)
):
    """
    Show random images from a dataframe.

    Args:
        df: DataFrame containing image paths and labels.
        image_col: Column containing full image paths.
        label_col: Column containing class labels.
        n: Number of images to show.
        seed: Random seed.
        cols: Number of columns in the image grid.
        figsize: Figure size.
    """
    sample_df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=figsize)

    for i, row in sample_df.iterrows():
        image = Image.open(row[image_col]).convert("RGB")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(f"Class: {row[label_col]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_one_image_per_class(
    df: pd.DataFrame,
    image_col: str = "full_path",
    label_col: str = "ClassId",
    max_classes: int = 16,
    seed: int = 42,
    cols: int = 4,
    figsize: tuple = (10, 10)
):
    """
    Show one random image from each selected class.

    Args:
        df: DataFrame containing image paths and labels.
        image_col: Column containing full image paths.
        label_col: Column containing class labels.
        max_classes: Maximum number of classes to display.
        seed: Random seed.
        cols: Number of columns in image grid.
        figsize: Figure size.
    """
    selected_classes = (
        df[label_col]
        .drop_duplicates()
        .sort_values()
        .head(max_classes)
        .tolist()
    )

    samples = []

    for class_id in selected_classes:
        class_sample = df[df[label_col] == class_id].sample(n=1, random_state=seed)
        samples.append(class_sample)

    sample_df = pd.concat(samples).reset_index(drop=True)
    rows = int(np.ceil(len(sample_df) / cols))

    plt.figure(figsize=figsize)

    for i, row in sample_df.iterrows():
        image = Image.open(row[image_col]).convert("RGB")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(f"Class: {row[label_col]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()




from sklearn.model_selection import StratifiedGroupKFold


def create_grouped_train_valid_split(
    df: pd.DataFrame,
    label_col: str = "ClassId",
    group_col: str = "track_id",
    n_splits: int = 5,
    seed: int = 42
):
    """
    Create a train/validation split using StratifiedGroupKFold.

    This tries to keep class distribution balanced while also making sure
    the same group does not appear in both train and validation.

    Args:
        df: Input dataframe.
        label_col: Target label column.
        group_col: Group column used to prevent leakage.
        n_splits: Number of folds. 5 means about 80/20 split.
        seed: Random seed.

    Returns:
        train_split_df, valid_split_df
    """
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    X = df
    y = df[label_col]
    groups = df[group_col]

    train_idx, valid_idx = next(splitter.split(X, y, groups))

    train_split_df = df.iloc[train_idx].reset_index(drop=True)
    valid_split_df = df.iloc[valid_idx].reset_index(drop=True)

    return train_split_df, valid_split_df


def check_split_quality(
    train_split_df: pd.DataFrame,
    valid_split_df: pd.DataFrame,
    label_col: str = "ClassId",
    group_col: str = "track_id"
) -> dict:
    """
    Create summary tables to check train/validation split quality.

    Args:
        train_split_df: Training split dataframe.
        valid_split_df: Validation split dataframe.
        label_col: Target label column.
        group_col: Group column.

    Returns:
        Dictionary of summary DataFrames.
    """
    train_groups = set(train_split_df[group_col].unique())
    valid_groups = set(valid_split_df[group_col].unique())
    overlap_groups = train_groups.intersection(valid_groups)

    split_overview = pd.DataFrame({
        "split": ["train", "valid"],
        "num_images": [len(train_split_df), len(valid_split_df)],
        "num_classes": [
            train_split_df[label_col].nunique(),
            valid_split_df[label_col].nunique()
        ],
        "num_groups": [
            train_split_df[group_col].nunique(),
            valid_split_df[group_col].nunique()
        ],
    })

    leakage_check = pd.DataFrame({
        "metric": ["overlapping_groups"],
        "value": [len(overlap_groups)]
    })

    train_class_counts = get_class_distribution(train_split_df, label_col=label_col)
    valid_class_counts = get_class_distribution(valid_split_df, label_col=label_col)

    class_balance = train_class_counts.merge(
        valid_class_counts,
        on=label_col,
        suffixes=("_train", "_valid")
    )

    class_balance["valid_ratio"] = (
        class_balance["image_count_valid"] /
        (class_balance["image_count_train"] + class_balance["image_count_valid"])
    )

    return {
        "split_overview": split_overview,
        "leakage_check": leakage_check,
        "split_class_balance": class_balance,
    }


from torchvision import transforms

from PIL import Image


def show_original_and_transformed_images(
    df: pd.DataFrame,
    transform,
    image_col: str = "full_path",
    label_col: str = "ClassId",
    n: int = 6,
    seed: int = 42,
    figsize: tuple = (12, 5)
):
    """
    Show original and transformed images side by side.

    This is only a visualization helper.
    The transform itself should be defined in the notebook.
    """
    sample_df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    plt.figure(figsize=figsize)

    for i, row in sample_df.iterrows():
        image = Image.open(row[image_col]).convert("RGB")
        transformed_image = transform(image)

        plt.subplot(2, n, i + 1)
        plt.imshow(image)
        plt.title(f"Original\nClass {row[label_col]}")
        plt.axis("off")

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(transformed_image.permute(1, 2, 0))
        plt.title(f"Transformed\n{tuple(transformed_image.shape)}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



def plot_training_curves(results: dict, figsize: tuple = (12, 4)):
    """
    Plot training and validation loss/accuracy curves.
    """
    epochs = range(1, len(results["train_loss"]) + 1)

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="Train loss")
    plt.plot(epochs, results["valid_loss"], label="Valid loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="Train accuracy")
    plt.plot(epochs, results["valid_acc"], label="Valid accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()



def get_model_predictions(model, dataloader, device):
    """
    Get predictions, true labels, and prediction probabilities from a model.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for prediction.
        device: Target device.

    Returns:
        y_true, y_pred, y_prob
    """
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)

            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.append(y.cpu())
            y_pred.append(preds.cpu())
            y_prob.append(probs.cpu())

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_prob = torch.cat(y_prob)

    return y_true, y_pred, y_prob


from sklearn.metrics import confusion_matrix


def create_confusion_matrix_df(y_true, y_pred, class_ids):
    """
    Create a confusion matrix as a pandas DataFrame.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_ids: List of class IDs.

    Returns:
        Confusion matrix DataFrame.
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{class_id}" for class_id in class_ids],
        columns=[f"pred_{class_id}" for class_id in class_ids]
    )

    return cm_df


def plot_confusion_matrix_df(cm_df, figsize=(14, 12)):
    """
    Plot a confusion matrix DataFrame.

    Args:
        cm_df: Confusion matrix DataFrame.
        figsize: Figure size.
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm_df.values)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(
        ticks=range(len(cm_df.columns)),
        labels=[col.replace("pred_", "") for col in cm_df.columns],
        rotation=90
    )
    plt.yticks(
        ticks=range(len(cm_df.index)),
        labels=[idx.replace("true_", "") for idx in cm_df.index]
    )
    plt.colorbar(label="Number of Images")
    plt.tight_layout()
    plt.show()


def get_most_confused_classes(cm_df, top_n=15):
    """
    Get the most common off-diagonal confusion pairs.

    Args:
        cm_df: Confusion matrix DataFrame.
        top_n: Number of confusion pairs to return.

    Returns:
        DataFrame with most confused class pairs.
    """
    confusion_records = []

    for true_label in cm_df.index:
        for pred_label in cm_df.columns:
            if true_label.replace("true_", "") != pred_label.replace("pred_", ""):
                count = cm_df.loc[true_label, pred_label]

                if count > 0:
                    confusion_records.append({
                        "true_class": true_label.replace("true_", ""),
                        "predicted_class": pred_label.replace("pred_", ""),
                        "count": count
                    })

    confused_df = pd.DataFrame(confusion_records)

    if len(confused_df) == 0:
        return confused_df

    confused_df = confused_df.sort_values("count", ascending=False).reset_index(drop=True)

    return confused_df.head(top_n)



from sklearn.metrics import classification_report


def create_classification_report_df(y_true, y_pred):
    """
    Create a classification report as a pandas DataFrame.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Classification report DataFrame.
    """
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "class_or_metric"})

    return report_df


def create_per_class_accuracy_df(y_true, y_pred):
    """
    Create per-class accuracy table.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        DataFrame with per-class accuracy.
    """
    y_true_series = pd.Series(y_true, name="true_class")
    y_pred_series = pd.Series(y_pred, name="predicted_class")

    eval_df = pd.concat([y_true_series, y_pred_series], axis=1)
    eval_df["correct"] = eval_df["true_class"] == eval_df["predicted_class"]

    per_class_acc = (
        eval_df
        .groupby("true_class")
        .agg(
            total_images=("correct", "count"),
            correct_predictions=("correct", "sum"),
            accuracy=("correct", "mean")
        )
        .reset_index()
        .sort_values("accuracy")
    )

    return per_class_acc


def show_wrong_predictions(
    wrong_df: pd.DataFrame,
    image_col: str = "full_path",
    true_col: str = "true_class",
    pred_col: str = "predicted_class",
    conf_col: str = "confidence",
    n: int = 16,
    sort_by_confidence: bool = True,
    ascending: bool = False,
    seed: int = 42,
    cols: int = 4,
    figsize: tuple = (12, 12)
):
    """
    Show wrong prediction images with true label, predicted label, and confidence.

    Args:
        wrong_df: DataFrame containing wrong predictions.
        image_col: Column containing image paths.
        true_col: Column containing true labels.
        pred_col: Column containing predicted labels.
        conf_col: Column containing prediction confidence.
        n: Number of images to show.
        sort_by_confidence: Whether to sort by confidence before selecting images.
        ascending: Sort order for confidence.
        seed: Random seed if not sorting by confidence.
        cols: Number of columns in the plot.
        figsize: Figure size.
    """
    if sort_by_confidence:
        sample_df = wrong_df.sort_values(conf_col, ascending=ascending).head(n).reset_index(drop=True)
    else:
        sample_df = wrong_df.sample(n=n, random_state=seed).reset_index(drop=True)

    rows = int(np.ceil(n / cols))

    plt.figure(figsize=figsize)

    for i, row in sample_df.iterrows():
        image = Image.open(row[image_col]).convert("RGB")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(
            f"True: {row[true_col]} | Pred: {row[pred_col]}\nConf: {row[conf_col]:.2f}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()