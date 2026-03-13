"""
Standalone autoencoder training script for cycle-level heart sound features.

Expected input files:
- outputs/{RUN_NAME}/preprocess/cycle_features.npy
- outputs/{RUN_NAME}/preprocess/cycle_metadata.csv
- outputs/{RUN_NAME}/preprocess/feature_names.json

Saved artifacts:
- outputs/{RUN_NAME}/training/autoencoder.keras
- outputs/{RUN_NAME}/training/encoder.keras
- outputs/{RUN_NAME}/training/model_summary.txt
- outputs/{RUN_NAME}/training/training_history.csv
- outputs/{RUN_NAME}/training/training_summary.json
- outputs/{RUN_NAME}/training/tensorboard/
- outputs/{RUN_NAME}/training/reconstruction_error_summary.csv
"""

from __future__ import annotations

import io
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from excel_export_utils import export_stage_workbook


# ============================================================================
# Editable configuration
# ============================================================================
class TrainConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent
    OUTPUT_ROOT = PROJECT_ROOT / "outputs"
    RUN_NAME = "test_dataset_260312_preprocess_v2"

    PREPROCESS_ROOT = OUTPUT_ROOT / RUN_NAME / "preprocess"
    TRAINING_ROOT = OUTPUT_ROOT / RUN_NAME / "training"
    REQUIRED_METADATA_COLUMNS = [
        "sample_id",
        "subject_id",
        "recording_id",
        "valid_flag",
    ]

    HIDDEN_UNITS = [64, 32]
    LATENT_DIM = 8
    ACTIVATION = "relu"
    OUTPUT_ACTIVATION = "linear"
    LOSS = "mse"
    OPTIMIZER_LEARNING_RATE = 1e-3

    VALIDATION_FRACTION = 0.20
    BATCH_SIZE = 64
    EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20
    TENSORBOARD_HISTOGRAM_FREQ = 0
    VERBOSE = 2

    EXCEL_EXPORT_ENABLED = True
    EXCEL_FILENAME = "training_data_export.xlsx"
    EXCEL_FREEZE_PANES = "A2"
    EXCEL_HEADER_FILL = "1F4E78"
    EXCEL_HEADER_FONT_COLOR = "FFFFFF"
    EXCEL_MAX_COLUMN_WIDTH = 40

    RANDOM_SEED = 42


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PATHS = {
    "output_root": TrainConfig.OUTPUT_ROOT,
}

RUN_NAME = TrainConfig.RUN_NAME

DATA = {
    "preprocess_root": TrainConfig.PREPROCESS_ROOT,
    "training_root": TrainConfig.TRAINING_ROOT,
    "required_metadata_columns": TrainConfig.REQUIRED_METADATA_COLUMNS,
}

PREPROCESS = {}

MODEL = {
    "hidden_units": TrainConfig.HIDDEN_UNITS,
    "latent_dim": TrainConfig.LATENT_DIM,
    "activation": TrainConfig.ACTIVATION,
    "output_activation": TrainConfig.OUTPUT_ACTIVATION,
    "loss": TrainConfig.LOSS,
    "optimizer_learning_rate": TrainConfig.OPTIMIZER_LEARNING_RATE,
}

TRAINING = {
    "validation_fraction": TrainConfig.VALIDATION_FRACTION,
    "batch_size": TrainConfig.BATCH_SIZE,
    "epochs": TrainConfig.EPOCHS,
    "early_stopping_patience": TrainConfig.EARLY_STOPPING_PATIENCE,
    "tensorboard_histogram_freq": TrainConfig.TENSORBOARD_HISTOGRAM_FREQ,
    "verbose": TrainConfig.VERBOSE,
}

EMBEDDING = {}

CLUSTERING = {}

EXCEL = {
    "export_enabled": TrainConfig.EXCEL_EXPORT_ENABLED,
    "filename": TrainConfig.EXCEL_FILENAME,
    "freeze_panes": TrainConfig.EXCEL_FREEZE_PANES,
    "header_fill": TrainConfig.EXCEL_HEADER_FILL,
    "header_font_color": TrainConfig.EXCEL_HEADER_FONT_COLOR,
    "max_column_width": TrainConfig.EXCEL_MAX_COLUMN_WIDTH,
}

RANDOM_SEED = TrainConfig.RANDOM_SEED


# ============================================================================
# Dataset adapter section
# ============================================================================
def load_preprocess_outputs(
    preprocess_root: Path,
    required_metadata_columns: list[str],
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Load the feature matrix and aligned valid-cycle metadata from preprocessing.

    Expected input schema:
    - cycle_features.npy: shape (num_valid_cycles, num_features)
    - cycle_metadata.csv: row-based metadata including valid_flag and sample_id
    - feature_names.json: one name per feature column

    Output:
        features: float32 array with shape (num_valid_cycles, num_features)
        valid_metadata: DataFrame aligned 1:1 with features
        feature_names: list with length num_features
    """
    feature_path = preprocess_root / "cycle_features.npy"
    metadata_path = preprocess_root / "cycle_metadata.csv"
    feature_names_path = preprocess_root / "feature_names.json"

    for path in [feature_path, metadata_path, feature_names_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required preprocess artifact is missing: {path}")

    features = np.load(feature_path).astype(np.float32)
    metadata = pd.read_csv(metadata_path)
    with open(feature_names_path, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    missing_columns = [
        column for column in required_metadata_columns if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required metadata columns: {missing_columns}")

    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Invalid feature matrix shape: {features.shape}")
    if len(feature_names) != features.shape[1]:
        raise ValueError(
            "Feature name count does not match feature matrix width: "
            f"{len(feature_names)} vs {features.shape[1]}"
        )
    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("Feature matrix contains NaN or Inf values.")

    valid_metadata = metadata.loc[metadata["valid_flag"] == True].copy()
    if valid_metadata.empty:
        raise ValueError("No valid cycles found in metadata.")
    if valid_metadata["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id must be unique across valid cycles.")

    if len(valid_metadata) != features.shape[0]:
        raise ValueError(
            "Number of valid metadata rows does not match feature rows: "
            f"{len(valid_metadata)} vs {features.shape[0]}"
        )

    # Prefer an explicit global feature index when available. This is the
    # safest alignment check because the feature matrix is saved row-wise.
    if "feature_row_index" in valid_metadata.columns:
        feature_row_index = valid_metadata["feature_row_index"].dropna().to_numpy()
        if (
            len(feature_row_index) == len(valid_metadata)
            and len(np.unique(feature_row_index)) == len(valid_metadata)
            and np.array_equal(
                np.sort(feature_row_index.astype(int)),
                np.arange(len(valid_metadata), dtype=int),
            )
        ):
            valid_metadata["feature_row_index"] = valid_metadata["feature_row_index"].astype(int)
            valid_metadata = valid_metadata.sort_values(
                by="feature_row_index", kind="stable"
            ).reset_index(drop=True)
            return features, valid_metadata, feature_names

    sort_columns = [
        column
        for column in ["recording_id", "cycle_index"]
        if column in valid_metadata.columns
    ]
    if not sort_columns:
        raise ValueError(
            "Unable to infer valid-cycle alignment. Expected feature_row_index or "
            "recording_id/cycle_index in metadata."
        )
    valid_metadata = valid_metadata.sort_values(
        by=sort_columns, kind="stable"
    ).reset_index(drop=True)

    return features, valid_metadata, feature_names


# ============================================================================
# Utility functions
# ============================================================================
def set_random_seed(seed: int) -> None:
    """Set reproducible random seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def ensure_output_directories(output_root: Path, run_name: str) -> dict[str, Path]:
    """Create stable output directories for this run."""
    run_root = output_root / run_name
    preprocess_root = run_root / "preprocess"
    training_root = run_root / "training"
    clustering_root = run_root / "clustering"
    interpretation_root = run_root / "interpretation"
    tensorboard_root = training_root / "tensorboard"

    for path in [
        run_root,
        preprocess_root,
        training_root,
        clustering_root,
        interpretation_root,
        tensorboard_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "run_root": run_root,
        "preprocess_root": preprocess_root,
        "training_root": training_root,
        "clustering_root": clustering_root,
        "interpretation_root": interpretation_root,
        "tensorboard_root": tensorboard_root,
    }


def choose_split_column(metadata: pd.DataFrame) -> str:
    """
    Select the group column used for train/validation splitting.

    Output:
        "subject_id" when available, otherwise "recording_id".
    """
    if "subject_id" in metadata.columns and metadata["subject_id"].notna().all():
        if metadata["subject_id"].astype(str).nunique() >= 2:
            return "subject_id"
    if "recording_id" not in metadata.columns:
        raise ValueError("Metadata must contain subject_id or recording_id for splitting.")
    if metadata["recording_id"].astype(str).nunique() < 2:
        raise ValueError("At least two unique groups are required for train/validation split.")
    return "recording_id"


def group_train_validation_split(
    metadata: pd.DataFrame,
    validation_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, str, list[str], list[str]]:
    """
    Split samples into train/validation at the group level.

    Input:
        metadata: DataFrame aligned to features with shape (num_samples, num_columns).
        validation_fraction: Fraction of groups assigned to validation.
    Output:
        train_mask: bool array with shape (num_samples,)
        validation_mask: bool array with shape (num_samples,)
        split_column: grouping column name
        train_groups: ordered group labels used for training
        validation_groups: ordered group labels used for validation
    """
    split_column = choose_split_column(metadata)
    groups = metadata[split_column].astype(str).drop_duplicates().tolist()
    if len(groups) < 2:
        raise ValueError("Need at least two unique groups for train/validation split.")

    rng = np.random.default_rng(random_seed)
    shuffled_groups = groups.copy()
    rng.shuffle(shuffled_groups)

    num_validation_groups = max(1, int(round(len(shuffled_groups) * validation_fraction)))
    num_validation_groups = min(num_validation_groups, len(shuffled_groups) - 1)
    validation_groups = shuffled_groups[:num_validation_groups]
    train_groups = shuffled_groups[num_validation_groups:]

    group_values = metadata[split_column].astype(str).to_numpy()
    validation_mask = np.isin(group_values, validation_groups)
    train_mask = ~validation_mask

    if train_mask.sum() == 0 or validation_mask.sum() == 0:
        raise ValueError("Train/validation split produced an empty split.")

    return train_mask, validation_mask, split_column, train_groups, validation_groups


def build_autoencoder(
    input_dim: int,
    hidden_units: list[int],
    latent_dim: int,
    activation: str,
    output_activation: str,
    learning_rate: float,
    normalizer: tf.keras.layers.Normalization,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Build a dense tabular autoencoder and its encoder view.

    Tensor shapes:
    - input tensor: (batch, input_dim)
    - latent tensor: (batch, latent_dim)
    - reconstruction tensor: (batch, input_dim)
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="cycle_features")
    x = normalizer(inputs)

    for layer_index, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(
            units,
            activation=activation,
            name=f"encoder_dense_{layer_index + 1}",
        )(x)

    latent = tf.keras.layers.Dense(
        latent_dim,
        activation="linear",
        name="latent_embedding",
    )(x)

    x = latent
    for layer_index, units in enumerate(reversed(hidden_units)):
        x = tf.keras.layers.Dense(
            units,
            activation=activation,
            name=f"decoder_dense_{layer_index + 1}",
        )(x)

    reconstruction = tf.keras.layers.Dense(
        input_dim,
        activation=output_activation,
        name="reconstruction",
    )(x)

    autoencoder = tf.keras.Model(inputs=inputs, outputs=reconstruction, name="autoencoder")
    encoder = tf.keras.Model(inputs=inputs, outputs=latent, name="encoder")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=MODEL["loss"],
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return autoencoder, encoder


def write_model_summary(model: tf.keras.Model, output_path: Path) -> None:
    """Save a text summary of a Keras model."""
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    output_path.write_text(buffer.getvalue(), encoding="utf-8")


def make_callbacks(training_root: Path, tensorboard_root: Path) -> list[tf.keras.callbacks.Callback]:
    """Build the standard callback set for training."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(training_root / "autoencoder.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=TRAINING["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(training_root / "training_history.csv")),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_root),
            histogram_freq=TRAINING["tensorboard_histogram_freq"],
        ),
    ]


def history_summary(history: tf.keras.callbacks.History) -> dict[str, float | int | None]:
    """Convert Keras history into a compact numeric summary."""
    loss_history = history.history.get("loss", [])
    val_loss_history = history.history.get("val_loss", [])
    return {
        "epochs_ran": len(loss_history),
        "best_validation_loss": float(np.min(val_loss_history)) if val_loss_history else None,
        "final_training_loss": float(loss_history[-1]) if loss_history else None,
        "final_validation_loss": float(val_loss_history[-1]) if val_loss_history else None,
    }


def reconstruction_error_table(
    autoencoder: tf.keras.Model,
    normalizer: tf.keras.layers.Normalization,
    features: np.ndarray,
    metadata: pd.DataFrame,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    batch_size: int,
) -> pd.DataFrame:
    """
    Compute per-sample reconstruction errors for all valid cycles.

    Input:
        features: raw feature matrix with shape (num_valid_cycles, input_dim)
        metadata: aligned valid metadata with shape (num_valid_cycles, num_columns)
    Output:
        DataFrame aligned 1:1 with valid cycles.
    """
    normalized_targets = normalizer(features).numpy().astype(np.float32)
    predictions = autoencoder.predict(features, batch_size=batch_size, verbose=0)

    squared_error = np.square(predictions - normalized_targets)
    absolute_error = np.abs(predictions - normalized_targets)
    reconstruction_mse = np.mean(squared_error, axis=1)
    reconstruction_mae = np.mean(absolute_error, axis=1)

    split_labels = np.where(train_mask, "train", "validation")
    if not np.all(train_mask | validation_mask):
        raise ValueError("Split masks do not cover every valid sample.")

    output = metadata[["sample_id", "subject_id", "recording_id"]].copy()
    if "feature_row_index" in metadata.columns:
        output["feature_row_index"] = metadata["feature_row_index"].astype(int)
    if "waveform_row_index" in metadata.columns:
        output["waveform_row_index"] = metadata["waveform_row_index"].astype(int)
    output["split"] = split_labels
    output["reconstruction_mse"] = reconstruction_mse
    output["reconstruction_mae"] = reconstruction_mae
    return output


def export_training_excel(
    training_root: Path,
    training_summary: dict[str, Any],
    history: tf.keras.callbacks.History,
    reconstruction_table: pd.DataFrame,
) -> Path:
    """Export training-stage artifacts to an Excel workbook."""
    overview_rows: list[dict[str, Any]] = []
    for key in [
        "run_name",
        "input_dimension",
        "num_training_samples",
        "num_validation_samples",
        "latent_dimension",
        "best_validation_loss",
        "final_training_loss",
        "final_validation_loss",
        "split_column",
        "tensorflow_version",
    ]:
        overview_rows.append({"section": "summary", "metric": key, "value": training_summary[key]})

    for key, value in training_summary["reconstruction_error"].items():
        overview_rows.append({"section": "reconstruction_error", "metric": key, "value": value})

    for key, value in training_summary["hyperparameters"].items():
        if isinstance(value, list):
            value = json.dumps(value)
        overview_rows.append({"section": "hyperparameters", "metric": key, "value": value})

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1, dtype=int))
    train_groups_df = pd.DataFrame({"train_group": training_summary["train_groups"]})
    validation_groups_df = pd.DataFrame({"validation_group": training_summary["validation_groups"]})

    workbook_path = training_root / EXCEL["filename"]
    return export_stage_workbook(
        workbook_path=workbook_path,
        sheets={
            "Overview": pd.DataFrame(overview_rows),
            "Training_History": history_df,
            "Reconstruction_Error": reconstruction_table,
            "Train_Groups": train_groups_df,
            "Validation_Groups": validation_groups_df,
        },
        freeze_panes=EXCEL["freeze_panes"],
        header_fill=EXCEL["header_fill"],
        header_font_color=EXCEL["header_font_color"],
        max_column_width=EXCEL["max_column_width"],
    )


def main() -> None:
    """Train the autoencoder on preprocessed cycle-level features."""
    set_random_seed(RANDOM_SEED)
    output_paths = ensure_output_directories(PATHS["output_root"], RUN_NAME)

    features, valid_metadata, feature_names = load_preprocess_outputs(
        preprocess_root=DATA["preprocess_root"],
        required_metadata_columns=DATA["required_metadata_columns"],
    )

    train_mask, validation_mask, split_column, train_groups, validation_groups = (
        group_train_validation_split(
            metadata=valid_metadata,
            validation_fraction=TRAINING["validation_fraction"],
            random_seed=RANDOM_SEED,
        )
    )

    x_train = features[train_mask]
    x_validation = features[validation_mask]
    input_dim = features.shape[1]
    batch_size = min(TRAINING["batch_size"], x_train.shape[0])

    if batch_size <= 0:
        raise ValueError("Training batch size resolved to zero.")

    normalizer = tf.keras.layers.Normalization(axis=-1, name="feature_normalization")
    normalizer.adapt(x_train)

    y_train = normalizer(x_train).numpy().astype(np.float32)
    y_validation = normalizer(x_validation).numpy().astype(np.float32)

    autoencoder, encoder = build_autoencoder(
        input_dim=input_dim,
        hidden_units=MODEL["hidden_units"],
        latent_dim=MODEL["latent_dim"],
        activation=MODEL["activation"],
        output_activation=MODEL["output_activation"],
        learning_rate=MODEL["optimizer_learning_rate"],
        normalizer=normalizer,
    )

    callbacks = make_callbacks(
        training_root=output_paths["training_root"],
        tensorboard_root=output_paths["tensorboard_root"],
    )

    history = autoencoder.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_validation, y_validation),
        epochs=TRAINING["epochs"],
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=TRAINING["verbose"],
    )

    autoencoder.save(output_paths["training_root"] / "autoencoder.keras")
    encoder.save(output_paths["training_root"] / "encoder.keras")
    write_model_summary(autoencoder, output_paths["training_root"] / "model_summary.txt")

    reconstruction_table = reconstruction_error_table(
        autoencoder=autoencoder,
        normalizer=normalizer,
        features=features,
        metadata=valid_metadata,
        train_mask=train_mask,
        validation_mask=validation_mask,
        batch_size=batch_size,
    )
    reconstruction_table.to_csv(
        output_paths["training_root"] / "reconstruction_error_summary.csv",
        index=False,
    )

    compact_history = history_summary(history)
    train_mean_mse = float(
        reconstruction_table.loc[
            reconstruction_table["split"] == "train", "reconstruction_mse"
        ].mean()
    )
    validation_mean_mse = float(
        reconstruction_table.loc[
            reconstruction_table["split"] == "validation", "reconstruction_mse"
        ].mean()
    )
    training_summary = {
        "run_name": RUN_NAME,
        "input_dimension": input_dim,
        "num_training_samples": int(x_train.shape[0]),
        "num_validation_samples": int(x_validation.shape[0]),
        "latent_dimension": MODEL["latent_dim"],
        "best_validation_loss": compact_history["best_validation_loss"],
        "final_training_loss": compact_history["final_training_loss"],
        "final_validation_loss": compact_history["final_validation_loss"],
        "feature_names_count": len(feature_names),
        "split_column": split_column,
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "tensorflow_version": tf.__version__,
        "hyperparameters": {
            "hidden_units": MODEL["hidden_units"],
            "activation": MODEL["activation"],
            "output_activation": MODEL["output_activation"],
            "loss": MODEL["loss"],
            "optimizer_learning_rate": MODEL["optimizer_learning_rate"],
            "validation_fraction": TRAINING["validation_fraction"],
            "batch_size": batch_size,
            "epochs": TRAINING["epochs"],
            "early_stopping_patience": TRAINING["early_stopping_patience"],
        },
        "history_summary": compact_history,
        "reconstruction_error": {
            "train_mean_mse": train_mean_mse,
            "validation_mean_mse": validation_mean_mse,
        },
    }

    with open(
        output_paths["training_root"] / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(training_summary, file, indent=2)

    excel_path = None
    if EXCEL["export_enabled"]:
        excel_path = export_training_excel(
            training_root=output_paths["training_root"],
            training_summary=training_summary,
            history=history,
            reconstruction_table=reconstruction_table,
        )

    logger.info("Saved training outputs to: %s", output_paths["training_root"])
    if excel_path is not None:
        logger.info("Saved training Excel export to: %s", excel_path)
    logger.info("Input dimension: %s", input_dim)
    logger.info("Training samples: %s", x_train.shape[0])
    logger.info("Validation samples: %s", x_validation.shape[0])
    logger.info("Best validation loss: %s", compact_history["best_validation_loss"])


if __name__ == "__main__":
    main()
