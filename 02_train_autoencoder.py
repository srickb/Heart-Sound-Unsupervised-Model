"""
Representation learning script for the revised Heart-Sound-Unsupervised-Model pipeline.

Expected input files:
- outputs/{PREPROCESS_RUN_NAME}/preprocess/beat_features_valid.csv
- outputs/{PREPROCESS_RUN_NAME}/preprocess/learning_input_columns.json
- outputs/{PREPROCESS_RUN_NAME}/preprocess/feature_names.json

Saved artifacts:
- outputs/{RUN_NAME}/training/dae_best.pt
- outputs/{RUN_NAME}/training/scaler.joblib
- outputs/{RUN_NAME}/training/training_history.csv
- outputs/{RUN_NAME}/training/reconstruction_summary_by_split.csv
- outputs/{RUN_NAME}/training/latent_train.csv
- outputs/{RUN_NAME}/training/latent_val.csv
- outputs/{RUN_NAME}/training/latent_test.csv
- outputs/{RUN_NAME}/training/latent_all_valid.csv
- outputs/{RUN_NAME}/training/split_info.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrainingConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent

    # 아래 2개 경로만 윈도우 절대경로로 직접 수정해서 사용하세요.
    PREPROCESS_INPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\preprocess"
    OUTPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\training"

    RANDOM_SEED = 42
    HIDDEN_DIMS = [256, 128, 64]
    LATENT_DIM = 12
    DROPOUT = 0.10
    BATCH_SIZE = 256
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    MASK_RATIO = 0.15
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    BEAT_FEATURES_FILENAME = "beat_features_valid.csv"
    LEARNING_INPUT_COLUMNS_FILENAME = "learning_input_columns.json"
    FEATURE_NAMES_FILENAME = "feature_names.json"

    MODEL_FILENAME = "dae_best.pt"
    SCALER_FILENAME = "scaler.joblib"
    TRAINING_HISTORY_FILENAME = "training_history.csv"
    RECONSTRUCTION_SUMMARY_FILENAME = "reconstruction_summary_by_split.csv"
    LATENT_TRAIN_FILENAME = "latent_train.csv"
    LATENT_VAL_FILENAME = "latent_val.csv"
    LATENT_TEST_FILENAME = "latent_test.csv"
    LATENT_ALL_VALID_FILENAME = "latent_all_valid.csv"
    SPLIT_INFO_FILENAME = "split_info.json"


REQUIRED_METADATA_COLUMNS = [
    "record_id",
    "beat_index",
    "valid_flag",
]

LATENT_METADATA_COLUMNS = [
    "record_id",
    "source_file",
    "beat_index",
    "cycle_index",
    "valid_flag",
]


@dataclass
class SplitData:
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    train_records: list[str]
    val_records: list[str]
    test_records: list[str]


# =================================================
# 1. Model Definition
# =================================================


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            dims=[input_dim, *hidden_dims, latent_dim],
            dropout=dropout,
            apply_dropout_on_last=False,
        )
        self.decoder = self._build_mlp(
            dims=[latent_dim, *list(reversed(hidden_dims)), input_dim],
            dropout=dropout,
            apply_dropout_on_last=False,
        )

    @staticmethod
    def _build_mlp(dims: list[int], dropout: float, apply_dropout_on_last: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            in_dim = dims[index]
            out_dim = dims[index + 1]
            is_last = index == len(dims) - 2
            layers.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            elif apply_dropout_on_last and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# =================================================
# 2. Utility Helpers
# =================================================


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configured_path(path_value: Path | str) -> Path:
    return Path(path_value).expanduser()


def ensure_output_directories(stage_output_folder: Path) -> dict[str, Path]:
    training_root = stage_output_folder
    training_root.mkdir(parents=True, exist_ok=True)
    return {"training_root": training_root}


def load_json_list(file_path: Path) -> list[str]:
    values = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Expected JSON string list: {file_path}")
    return values


def _safe_split_ratio_sum(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = float(train_ratio + val_ratio + test_ratio)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


# =================================================
# 3. Input Loading / Validation
# =================================================


def load_representation_inputs(input_root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    logger.info("입력 파일 로드 시작: %s", input_root)
    beat_features_path = input_root / TrainingConfig.BEAT_FEATURES_FILENAME
    learning_input_columns_path = input_root / TrainingConfig.LEARNING_INPUT_COLUMNS_FILENAME
    feature_names_path = input_root / TrainingConfig.FEATURE_NAMES_FILENAME

    missing_paths = [
        path for path in [beat_features_path, learning_input_columns_path, feature_names_path] if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing representation learning inputs: {missing_paths}")

    feature_frame = pd.read_csv(beat_features_path)
    learning_input_columns = load_json_list(learning_input_columns_path)
    feature_names = load_json_list(feature_names_path)
    logger.info("입력 파일 로드 완료: num_valid_beats=%s", len(feature_frame))
    return feature_frame, learning_input_columns, feature_names


def validate_input_frame(
    feature_frame: pd.DataFrame,
    learning_input_columns: list[str],
    feature_names: list[str],
) -> None:
    missing_metadata = [column for column in REQUIRED_METADATA_COLUMNS if column not in feature_frame.columns]
    if missing_metadata:
        raise ValueError(f"Missing required metadata columns: {missing_metadata}")

    missing_learning_columns = [column for column in learning_input_columns if column not in feature_frame.columns]
    if missing_learning_columns:
        raise ValueError(f"Missing learning input columns in beat_features_valid.csv: {missing_learning_columns}")

    unknown_learning_columns = [column for column in learning_input_columns if column not in feature_names]
    if unknown_learning_columns:
        raise ValueError(f"Columns in learning_input_columns.json are absent from feature_names.json: {unknown_learning_columns}")

    valid_flags = pd.to_numeric(feature_frame["valid_flag"], errors="raise").astype(np.int64)
    if np.any(valid_flags != 1):
        raise ValueError("beat_features_valid.csv must contain only valid beats with valid_flag = 1")

    if feature_frame[learning_input_columns].isnull().any().any():
        null_counts = feature_frame[learning_input_columns].isnull().sum()
        failing = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Learning input columns contain null values: {failing}")

    numeric_matrix = feature_frame.loc[:, learning_input_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(numeric_matrix).all():
        raise ValueError("Learning input matrix contains non-finite values")


# =================================================
# 4. Split Helpers
# =================================================


def allocate_split_counts(num_records: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    if num_records < 3:
        raise ValueError("At least three distinct record_id values are required for train/validation/test split")

    _safe_split_ratio_sum(train_ratio, val_ratio, test_ratio)
    ratios = {
        "train": train_ratio,
        "validation": val_ratio,
        "test": test_ratio,
    }
    raw_counts = {name: num_records * ratio for name, ratio in ratios.items()}
    counts = {name: int(np.floor(value)) for name, value in raw_counts.items()}
    remainder = num_records - sum(counts.values())

    fractional_priority = sorted(
        ratios.keys(),
        key=lambda name: (raw_counts[name] - counts[name], ratios[name]),
        reverse=True,
    )
    for split_name in fractional_priority[:remainder]:
        counts[split_name] += 1

    for split_name in ["validation", "test", "train"]:
        if counts[split_name] > 0:
            continue
        donor = max(
            (name for name in counts if counts[name] > 1),
            key=lambda name: counts[name],
            default=None,
        )
        if donor is None:
            raise ValueError("Unable to allocate non-empty train/validation/test split")
        counts[donor] -= 1
        counts[split_name] += 1

    if sum(counts.values()) != num_records:
        raise ValueError("Split count allocation failed")

    return counts


def build_record_group_split(
    feature_frame: pd.DataFrame,
    random_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> SplitData:
    record_ids = feature_frame["record_id"].drop_duplicates().astype(str).tolist()
    rng = np.random.default_rng(random_seed)
    shuffled_records = list(record_ids)
    rng.shuffle(shuffled_records)

    split_counts = allocate_split_counts(
        num_records=len(shuffled_records),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_end = split_counts["train"]
    val_end = train_end + split_counts["validation"]

    train_records = shuffled_records[:train_end]
    val_records = shuffled_records[train_end:val_end]
    test_records = shuffled_records[val_end:]

    train_frame = feature_frame.loc[feature_frame["record_id"].isin(train_records)].copy()
    val_frame = feature_frame.loc[feature_frame["record_id"].isin(val_records)].copy()
    test_frame = feature_frame.loc[feature_frame["record_id"].isin(test_records)].copy()

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError("Train/validation/test split must each contain at least one beat")

    return SplitData(
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
    )


def dataframe_to_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    matrix = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional")
    return matrix


# =================================================
# 5. Training Helpers
# =================================================


def build_tensor_loader(matrix: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(matrix))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def mask_numpy_array(matrix: np.ndarray, mask_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(matrix.shape, dtype=np.float32) < mask_ratio
    masked = matrix.copy()
    masked[mask] = 0.0
    return masked.astype(np.float32)


def mask_tensor(inputs: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    mask = torch.rand(inputs.shape, device=inputs.device) < mask_ratio
    return torch.where(mask, torch.zeros_like(inputs), inputs)


def evaluate_reconstruction_loss(
    model: DenoisingAutoencoder,
    input_matrix: np.ndarray,
    target_matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(input_matrix), torch.from_numpy(target_matrix))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for input_batch, target_batch in loader:
            input_batch = input_batch.to(device=device, dtype=torch.float32)
            target_batch = target_batch.to(device=device, dtype=torch.float32)
            outputs = model(input_batch)
            loss = criterion(outputs, target_batch)
            batch_size_current = int(target_batch.shape[0])
            total_loss += float(loss.item()) * batch_size_current
            total_count += batch_size_current

    if total_count == 0:
        raise ValueError("Evaluation loader is empty")

    return total_loss / float(total_count)


def train_denoising_autoencoder(
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    input_dim: int,
    training_root: Path,
    config: TrainingConfig,
) -> tuple[DenoisingAutoencoder, pd.DataFrame, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder(
        input_dim=input_dim,
        hidden_dims=list(config.HIDDEN_DIMS),
        latent_dim=config.LATENT_DIM,
        dropout=config.DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    train_loader = build_tensor_loader(
        matrix=train_matrix,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED,
    )
    val_noisy = mask_numpy_array(val_matrix, mask_ratio=config.MASK_RATIO, seed=config.RANDOM_SEED)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict: dict[str, torch.Tensor] | None = None
    history_rows: list[dict[str, float | int]] = []
    patience_counter = 0

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for (clean_batch,) in train_loader:
            clean_batch = clean_batch.to(device=device, dtype=torch.float32)
            noisy_batch = mask_tensor(clean_batch, mask_ratio=config.MASK_RATIO)

            optimizer.zero_grad()
            output_batch = model(noisy_batch)
            loss = criterion(output_batch, clean_batch)
            loss.backward()
            optimizer.step()

            batch_size_current = int(clean_batch.shape[0])
            train_loss_sum += float(loss.item()) * batch_size_current
            train_count += batch_size_current

        train_loss = train_loss_sum / float(train_count)
        val_loss = evaluate_reconstruction_loss(
            model=model,
            input_matrix=val_noisy,
            target_matrix=val_matrix,
            batch_size=config.BATCH_SIZE,
            device=device,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        logger.info("epoch=%s train_loss=%.8f val_loss=%.8f", epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            break

    if best_state_dict is None or best_epoch < 0:
        raise RuntimeError("Best model checkpoint was not created")

    model.load_state_dict(best_state_dict)
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "input_dim": input_dim,
            "hidden_dims": list(config.HIDDEN_DIMS),
            "latent_dim": config.LATENT_DIM,
            "dropout": config.DROPOUT,
            "mask_ratio": config.MASK_RATIO,
            "random_seed": config.RANDOM_SEED,
        },
        training_root / config.MODEL_FILENAME,
    )
    logger.info("best epoch: %s", best_epoch)

    history_frame = pd.DataFrame(history_rows)
    return model, history_frame, best_epoch


# =================================================
# 6. Latent / Summary Export
# =================================================


def extract_latent_dataframe(
    model: DenoisingAutoencoder,
    feature_frame: pd.DataFrame,
    scaled_matrix: np.ndarray,
    batch_size: int,
) -> pd.DataFrame:
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.from_numpy(scaled_matrix)), batch_size=batch_size, shuffle=False)

    latent_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            latent = model.encode(batch).cpu().numpy().astype(np.float32)
            latent_batches.append(latent)

    latent_matrix = np.vstack(latent_batches)
    latent_columns = [f"latent_{index:02d}" for index in range(latent_matrix.shape[1])]

    metadata_columns = [column for column in LATENT_METADATA_COLUMNS if column in feature_frame.columns]
    latent_frame = feature_frame.loc[:, metadata_columns].copy()
    for column_index, column_name in enumerate(latent_columns):
        latent_frame[column_name] = latent_matrix[:, column_index]
    return latent_frame


def build_reconstruction_summary(
    model: DenoisingAutoencoder,
    split_frames: dict[str, pd.DataFrame],
    split_matrices: dict[str, np.ndarray],
    batch_size: int,
    mask_ratio: float,
    random_seed: int,
) -> pd.DataFrame:
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []

    for split_index, split_name in enumerate(["train", "val", "test", "all_valid"], start=1):
        frame = split_frames[split_name]
        matrix = split_matrices[split_name]
        masked_matrix = mask_numpy_array(
            matrix=matrix,
            mask_ratio=mask_ratio,
            seed=random_seed + split_index,
        )
        clean_loss = evaluate_reconstruction_loss(
            model=model,
            input_matrix=matrix,
            target_matrix=matrix,
            batch_size=batch_size,
            device=device,
        )
        masked_loss = evaluate_reconstruction_loss(
            model=model,
            input_matrix=masked_matrix,
            target_matrix=matrix,
            batch_size=batch_size,
            device=device,
        )
        rows.append(
            {
                "split": split_name,
                "num_records": int(frame["record_id"].nunique()),
                "num_beats": int(len(frame)),
                "reconstruction_loss_clean": clean_loss,
                "reconstruction_loss_masked": masked_loss,
            }
        )

    return pd.DataFrame(rows)


def save_split_info(split_data: SplitData, training_root: Path, config: TrainingConfig, input_dim: int) -> None:
    split_info = {
        "train_records": split_data.train_records,
        "validation_records": split_data.val_records,
        "test_records": split_data.test_records,
        "random_seed": config.RANDOM_SEED,
        "input_dim": int(input_dim),
        "hidden_dims": list(config.HIDDEN_DIMS),
        "latent_dim": int(config.LATENT_DIM),
        "dropout": float(config.DROPOUT),
        "mask_ratio": float(config.MASK_RATIO),
    }
    (training_root / config.SPLIT_INFO_FILENAME).write_text(
        json.dumps(split_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# =================================================
# 7. Main
# =================================================


def main() -> None:
    config = TrainingConfig()
    set_random_seed(config.RANDOM_SEED)

    output_paths = ensure_output_directories(stage_output_folder=configured_path(config.OUTPUT_FOLDER))
    training_root = output_paths["training_root"]

    preprocess_input_root = configured_path(config.PREPROCESS_INPUT_FOLDER)
    logger.info("전처리 입력 폴더: %s", preprocess_input_root)
    logger.info("학습 출력 폴더: %s", training_root)

    feature_frame, learning_input_columns, feature_names = load_representation_inputs(preprocess_input_root)
    validate_input_frame(
        feature_frame=feature_frame,
        learning_input_columns=learning_input_columns,
        feature_names=feature_names,
    )
    logger.info("전체 valid beat 수: %s", len(feature_frame))

    split_data = build_record_group_split(
        feature_frame=feature_frame,
        random_seed=config.RANDOM_SEED,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
    )
    logger.info(
        "train / validation / test record 수: %s / %s / %s",
        len(split_data.train_records),
        len(split_data.val_records),
        len(split_data.test_records),
    )
    logger.info(
        "train / validation / test beat 수: %s / %s / %s",
        len(split_data.train_frame),
        len(split_data.val_frame),
        len(split_data.test_frame),
    )

    train_matrix = dataframe_to_feature_matrix(split_data.train_frame, learning_input_columns)
    val_matrix = dataframe_to_feature_matrix(split_data.val_frame, learning_input_columns)
    test_matrix = dataframe_to_feature_matrix(split_data.test_frame, learning_input_columns)
    all_valid_matrix = dataframe_to_feature_matrix(feature_frame, learning_input_columns)
    input_dim = int(train_matrix.shape[1])
    logger.info("input dimension: %s", input_dim)

    scaler = RobustScaler()
    scaler.fit(train_matrix)
    joblib.dump(scaler, training_root / config.SCALER_FILENAME)
    logger.info("train split 기준 scaler fit 완료")

    train_scaled = scaler.transform(train_matrix).astype(np.float32)
    val_scaled = scaler.transform(val_matrix).astype(np.float32)
    test_scaled = scaler.transform(test_matrix).astype(np.float32)
    all_valid_scaled = scaler.transform(all_valid_matrix).astype(np.float32)

    model, history_frame, best_epoch = train_denoising_autoencoder(
        train_matrix=train_scaled,
        val_matrix=val_scaled,
        input_dim=input_dim,
        training_root=training_root,
        config=config,
    )
    history_frame.to_csv(training_root / config.TRAINING_HISTORY_FILENAME, index=False)

    latent_train = extract_latent_dataframe(
        model=model,
        feature_frame=split_data.train_frame,
        scaled_matrix=train_scaled,
        batch_size=config.BATCH_SIZE,
    )
    latent_val = extract_latent_dataframe(
        model=model,
        feature_frame=split_data.val_frame,
        scaled_matrix=val_scaled,
        batch_size=config.BATCH_SIZE,
    )
    latent_test = extract_latent_dataframe(
        model=model,
        feature_frame=split_data.test_frame,
        scaled_matrix=test_scaled,
        batch_size=config.BATCH_SIZE,
    )
    latent_all_valid = extract_latent_dataframe(
        model=model,
        feature_frame=feature_frame,
        scaled_matrix=all_valid_scaled,
        batch_size=config.BATCH_SIZE,
    )

    latent_train.to_csv(training_root / config.LATENT_TRAIN_FILENAME, index=False)
    latent_val.to_csv(training_root / config.LATENT_VAL_FILENAME, index=False)
    latent_test.to_csv(training_root / config.LATENT_TEST_FILENAME, index=False)
    latent_all_valid.to_csv(training_root / config.LATENT_ALL_VALID_FILENAME, index=False)

    reconstruction_summary = build_reconstruction_summary(
        model=model,
        split_frames={
            "train": split_data.train_frame,
            "val": split_data.val_frame,
            "test": split_data.test_frame,
            "all_valid": feature_frame,
        },
        split_matrices={
            "train": train_scaled,
            "val": val_scaled,
            "test": test_scaled,
            "all_valid": all_valid_scaled,
        },
        batch_size=config.BATCH_SIZE,
        mask_ratio=config.MASK_RATIO,
        random_seed=config.RANDOM_SEED,
    )
    reconstruction_summary.to_csv(training_root / config.RECONSTRUCTION_SUMMARY_FILENAME, index=False)

    save_split_info(
        split_data=split_data,
        training_root=training_root,
        config=config,
        input_dim=input_dim,
    )

    logger.info("best epoch=%s", best_epoch)
    logger.info("latent export 완료: %s", training_root)


if __name__ == "__main__":
    main()
