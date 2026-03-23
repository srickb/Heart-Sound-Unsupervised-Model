"""
Cluster training script fixed to the current unsupervised design.

Expected input files:
- outputs/beat_level_preprocess_fixed/preprocess/beat_features_valid.csv
- outputs/beat_level_preprocess_fixed/preprocess/learning_input_columns.json
- outputs/representation_learning_fixed/training/latent_valid.csv
- outputs/representation_learning_fixed/training/dae_best.pt
- outputs/representation_learning_fixed/training/scaler.joblib
- outputs/representation_learning_fixed/training/split_info.json

Saved artifacts:
- outputs/{RUN_NAME}/clustering/idec_best.pt
- outputs/{RUN_NAME}/clustering/cluster_centers.npy
- outputs/{RUN_NAME}/clustering/cluster_assignments_valid.csv
- outputs/{RUN_NAME}/clustering/cluster_training_history.csv
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ClusteringConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent

    # 아래 3개 경로만 윈도우 절대경로로 직접 수정해서 사용하세요.
    PREPROCESS_INPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\preprocess"
    TRAINING_INPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\training"
    OUTPUT_FOLDER = r"C:\Users\LUI\Desktop\PCG\processed_data\260310\clustering"
    RANDOM_SEED = 42
    NUM_CLUSTERS = 4
    LATENT_DIM = 12
    BATCH_SIZE = 256
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    CLUSTER_LOSS_WEIGHT = 0.1

    BEAT_FEATURES_FILENAME = "beat_features_valid.csv"
    LEARNING_INPUT_COLUMNS_FILENAME = "learning_input_columns.json"
    LATENT_FILENAME = "latent_valid.csv"
    DAE_CHECKPOINT_FILENAME = "dae_best.pt"
    SCALER_FILENAME = "scaler.joblib"
    SPLIT_INFO_FILENAME = "split_info.json"

    IDEC_CHECKPOINT_FILENAME = "idec_best.pt"
    CLUSTER_CENTERS_FILENAME = "cluster_centers.npy"
    ASSIGNMENTS_FILENAME = "cluster_assignments_valid.csv"
    HISTORY_FILENAME = "cluster_training_history.csv"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


REQUIRED_METADATA_COLUMNS = [
    "record_id",
    "beat_index",
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


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return latent, decoded


class IDECModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_clusters: int) -> None:
        super().__init__()
        self.autoencoder = DenoisingAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.cluster_centers = nn.Parameter(torch.empty(num_clusters, latent_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, decoded = self.autoencoder(x)
        q = student_t_soft_assignment(latent, self.cluster_centers)
        return latent, decoded, q


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configured_path(path_value: Path | str) -> Path:
    return Path(path_value).expanduser()


def ensure_output_directories(stage_output_folder: Path) -> dict[str, Path]:
    clustering_root = stage_output_folder
    clustering_root.mkdir(parents=True, exist_ok=True)
    return {"clustering_root": clustering_root}


def load_json_list(file_path: Path) -> list[str]:
    values = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Expected JSON string list: {file_path}")
    return values


def load_split_info(file_path: Path) -> dict[str, object]:
    split_info = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(split_info, dict):
        raise ValueError(f"Expected JSON object: {file_path}")
    return split_info


def load_clustering_inputs(
    preprocess_root: Path,
    representation_root: Path,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict[str, object], object, dict[str, object]]:
    logger.info("입력 파일 로드 시작")

    beat_features_path = preprocess_root / ClusteringConfig.BEAT_FEATURES_FILENAME
    learning_input_columns_path = preprocess_root / ClusteringConfig.LEARNING_INPUT_COLUMNS_FILENAME
    latent_path = representation_root / ClusteringConfig.LATENT_FILENAME
    dae_checkpoint_path = representation_root / ClusteringConfig.DAE_CHECKPOINT_FILENAME
    scaler_path = representation_root / ClusteringConfig.SCALER_FILENAME
    split_info_path = representation_root / ClusteringConfig.SPLIT_INFO_FILENAME

    missing_paths = [
        path
        for path in [
            beat_features_path,
            learning_input_columns_path,
            latent_path,
            dae_checkpoint_path,
            scaler_path,
            split_info_path,
        ]
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing clustering inputs: {missing_paths}")

    beat_features = pd.read_csv(beat_features_path)
    learning_input_columns = load_json_list(learning_input_columns_path)
    latent_frame = pd.read_csv(latent_path)
    scaler = joblib.load(scaler_path)
    checkpoint = torch.load(dae_checkpoint_path, map_location="cpu")
    split_info = load_split_info(split_info_path)

    logger.info("scaler 로드 완료")
    logger.info("split 정보 로드 완료")
    logger.info("pretrained DAE 로드 완료")
    return beat_features, learning_input_columns, latent_frame, split_info, scaler, checkpoint


def validate_feature_inputs(
    beat_features: pd.DataFrame,
    learning_input_columns: list[str],
    latent_frame: pd.DataFrame,
) -> None:
    missing_metadata = [column for column in REQUIRED_METADATA_COLUMNS if column not in beat_features.columns]
    if missing_metadata:
        raise ValueError(f"Missing required metadata columns: {missing_metadata}")

    missing_learning_columns = [column for column in learning_input_columns if column not in beat_features.columns]
    if missing_learning_columns:
        raise ValueError(f"Missing learning input columns in beat_features_valid.csv: {missing_learning_columns}")

    valid_flags = pd.to_numeric(beat_features["valid_flag"], errors="raise").astype(np.int64)
    if np.any(valid_flags != 1):
        raise ValueError("beat_features_valid.csv must contain only valid beats with valid_flag = 1")

    if beat_features[learning_input_columns].isnull().any().any():
        null_counts = beat_features[learning_input_columns].isnull().sum()
        failing = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Learning input columns contain null values: {failing}")

    latent_columns = [f"latent_{index:02d}" for index in range(ClusteringConfig.LATENT_DIM)]
    missing_latent_columns = [column for column in ["record_id", "beat_index", *latent_columns] if column not in latent_frame.columns]
    if missing_latent_columns:
        raise ValueError(f"Missing latent columns in latent_valid.csv: {missing_latent_columns}")

    beat_index_left = beat_features.loc[:, ["record_id", "beat_index"]].copy()
    beat_index_right = latent_frame.loc[:, ["record_id", "beat_index"]].copy()
    if len(beat_index_left) != len(beat_index_right):
        raise ValueError("beat_features_valid.csv and latent_valid.csv row counts do not match")

    if not beat_index_left.reset_index(drop=True).equals(beat_index_right.reset_index(drop=True)):
        raise ValueError("beat_features_valid.csv and latent_valid.csv metadata order does not match")


def build_split_frames(feature_frame: pd.DataFrame, split_info: dict[str, object]) -> SplitData:
    train_records = [str(value) for value in split_info.get("train_records", [])]
    val_records = [str(value) for value in split_info.get("validation_records", [])]
    test_records = [str(value) for value in split_info.get("test_records", [])]

    if not train_records or not val_records or not test_records:
        raise ValueError("split_info.json must contain non-empty train_records, validation_records, and test_records")

    train_frame = feature_frame.loc[feature_frame["record_id"].astype(str).isin(train_records)].copy()
    val_frame = feature_frame.loc[feature_frame["record_id"].astype(str).isin(val_records)].copy()
    test_frame = feature_frame.loc[feature_frame["record_id"].astype(str).isin(test_records)].copy()

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


def student_t_soft_assignment(latent: torch.Tensor, cluster_centers: torch.Tensor) -> torch.Tensor:
    squared_distance = torch.sum((latent.unsqueeze(1) - cluster_centers.unsqueeze(0)) ** 2, dim=2)
    numerator = 1.0 / (1.0 + squared_distance)
    power = (1.0 + 1.0) / 2.0
    numerator = numerator.pow(power)
    return numerator / torch.sum(numerator, dim=1, keepdim=True)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
    return weight / torch.sum(weight, dim=1, keepdim=True)


def extract_latent_numpy(
    encoder: nn.Module,
    matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(matrix)), batch_size=batch_size, shuffle=False)
    latent_batches: list[np.ndarray] = []

    encoder.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            latent = encoder(batch).cpu().numpy().astype(np.float32)
            latent_batches.append(latent)

    return np.vstack(latent_batches)


def initialize_cluster_centers(train_latent: np.ndarray, num_clusters: int, random_seed: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init=20)
    kmeans.fit(train_latent)
    logger.info("KMeans 초기화 완료")
    return kmeans.cluster_centers_.astype(np.float32)


def compute_epoch_target_distribution(
    model: IDECModel,
    train_matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(train_matrix)), batch_size=batch_size, shuffle=False)
    q_batches: list[np.ndarray] = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            _latent, _decoded, q = model(batch)
            q_batches.append(q.cpu().numpy().astype(np.float32))

    q_all = np.vstack(q_batches)
    q_tensor = torch.from_numpy(q_all)
    p_all = target_distribution(q_tensor).numpy().astype(np.float32)
    return p_all


def build_training_loader(
    clean_matrix: np.ndarray,
    target_matrix: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(clean_matrix), torch.from_numpy(target_matrix))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def evaluate_idec(
    model: IDECModel,
    matrix: np.ndarray,
    batch_size: int,
    cluster_loss_weight: float,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(matrix)), batch_size=batch_size, shuffle=False)
    mse_loss = nn.MSELoss(reduction="mean")
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    total_recon = 0.0
    total_cluster = 0.0
    total_count = 0
    q_batches: list[np.ndarray] = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            _latent, decoded, q = model(batch)
            q_batches.append(q.cpu().numpy().astype(np.float32))
            recon = mse_loss(decoded, batch)
            batch_size_current = int(batch.shape[0])
            total_recon += float(recon.item()) * batch_size_current
            total_count += batch_size_current

    q_all = np.vstack(q_batches)
    p_all = target_distribution(torch.from_numpy(q_all)).to(device=device, dtype=torch.float32)
    q_all_tensor = torch.from_numpy(q_all).to(device=device, dtype=torch.float32)
    cluster = kl_loss(torch.log(q_all_tensor + 1e-12), p_all)

    recon_loss = total_recon / float(total_count)
    cluster_loss = float(cluster.item())
    total_loss = recon_loss + cluster_loss_weight * cluster_loss
    return recon_loss, cluster_loss, total_loss


def train_idec(
    train_matrix: np.ndarray,
    val_matrix: np.ndarray,
    dae_checkpoint: dict[str, object],
    input_dim: int,
    config: ClusteringConfig,
    clustering_root: Path,
) -> tuple[IDECModel, pd.DataFrame, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDECModel(
        input_dim=input_dim,
        latent_dim=config.LATENT_DIM,
        num_clusters=config.NUM_CLUSTERS,
    ).to(device)

    model.autoencoder.load_state_dict(dae_checkpoint["model_state_dict"])

    train_latent = extract_latent_numpy(
        encoder=model.autoencoder.encoder,
        matrix=train_matrix,
        batch_size=config.BATCH_SIZE,
        device=device,
    )
    logger.info("train latent 추출 완료")
    initial_centers = initialize_cluster_centers(
        train_latent=train_latent,
        num_clusters=config.NUM_CLUSTERS,
        random_seed=config.RANDOM_SEED,
    )
    model.cluster_centers.data.copy_(torch.from_numpy(initial_centers).to(device=device, dtype=torch.float32))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    mse_loss = nn.MSELoss(reduction="mean")
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    best_total_loss = float("inf")
    best_epoch = -1
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_centers: np.ndarray | None = None
    patience_counter = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_targets = compute_epoch_target_distribution(
            model=model,
            train_matrix=train_matrix,
            batch_size=config.BATCH_SIZE,
            device=device,
        )
        train_loader = build_training_loader(
            clean_matrix=train_matrix,
            target_matrix=train_targets,
            batch_size=config.BATCH_SIZE,
            seed=config.RANDOM_SEED + epoch,
        )

        model.train()
        train_recon_sum = 0.0
        train_cluster_sum = 0.0
        train_count = 0

        for clean_batch, target_batch in train_loader:
            clean_batch = clean_batch.to(device=device, dtype=torch.float32)
            target_batch = target_batch.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            _latent, decoded, q = model(clean_batch)
            recon = mse_loss(decoded, clean_batch)
            cluster = kl_loss(torch.log(q + 1e-12), target_batch)
            total = recon + config.CLUSTER_LOSS_WEIGHT * cluster
            total.backward()
            optimizer.step()

            batch_size_current = int(clean_batch.shape[0])
            train_recon_sum += float(recon.item()) * batch_size_current
            train_cluster_sum += float(cluster.item()) * batch_size_current
            train_count += batch_size_current

        train_recon_loss = train_recon_sum / float(train_count)
        train_cluster_loss = train_cluster_sum / float(train_count)
        train_total_loss = train_recon_loss + config.CLUSTER_LOSS_WEIGHT * train_cluster_loss

        val_recon_loss, val_cluster_loss, val_total_loss = evaluate_idec(
            model=model,
            matrix=val_matrix,
            batch_size=config.BATCH_SIZE,
            cluster_loss_weight=config.CLUSTER_LOSS_WEIGHT,
            device=device,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_recon_loss": train_recon_loss,
                "train_cluster_loss": train_cluster_loss,
                "train_total_loss": train_total_loss,
                "val_recon_loss": val_recon_loss,
                "val_cluster_loss": val_cluster_loss,
                "val_total_loss": val_total_loss,
            }
        )
        logger.info(
            "epoch=%s train_recon_loss=%.8f train_cluster_loss=%.8f train_total_loss=%.8f "
            "val_recon_loss=%.8f val_cluster_loss=%.8f val_total_loss=%.8f",
            epoch,
            train_recon_loss,
            train_cluster_loss,
            train_total_loss,
            val_recon_loss,
            val_cluster_loss,
            val_total_loss,
        )

        if val_total_loss < best_total_loss:
            best_total_loss = val_total_loss
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_centers = model.cluster_centers.detach().cpu().numpy().astype(np.float32)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            break

    if best_state_dict is None or best_centers is None or best_epoch < 0:
        raise RuntimeError("Best IDEC checkpoint was not created")

    model.load_state_dict(best_state_dict)
    checkpoint_path = clustering_root / config.IDEC_CHECKPOINT_FILENAME
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "input_dim": input_dim,
            "latent_dim": config.LATENT_DIM,
            "num_clusters": config.NUM_CLUSTERS,
            "cluster_loss_weight": config.CLUSTER_LOSS_WEIGHT,
            "random_seed": config.RANDOM_SEED,
        },
        checkpoint_path,
    )
    logger.info("best epoch: %s", best_epoch)

    history_frame = pd.DataFrame(history_rows)
    return model, history_frame, best_centers


def infer_assignments(
    model: IDECModel,
    feature_frame: pd.DataFrame,
    matrix: np.ndarray,
    batch_size: int,
) -> pd.DataFrame:
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.from_numpy(matrix)), batch_size=batch_size, shuffle=False)

    latent_batches: list[np.ndarray] = []
    q_batches: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device=device, dtype=torch.float32)
            latent, _decoded, q = model(batch)
            latent_batches.append(latent.cpu().numpy().astype(np.float32))
            q_batches.append(q.cpu().numpy().astype(np.float32))

    latent_matrix = np.vstack(latent_batches)
    q_matrix = np.vstack(q_batches)
    cluster_labels = np.argmax(q_matrix, axis=1).astype(np.int64)
    cluster_confidence = np.max(q_matrix, axis=1).astype(np.float32)

    assignment_frame = feature_frame.loc[:, ["record_id", "beat_index"]].copy()
    assignment_frame["cluster_label"] = cluster_labels
    assignment_frame["cluster_confidence"] = cluster_confidence
    for cluster_index in range(q_matrix.shape[1]):
        assignment_frame[f"q_cluster_{cluster_index}"] = q_matrix[:, cluster_index]
    for latent_index in range(latent_matrix.shape[1]):
        assignment_frame[f"latent_{latent_index:02d}"] = latent_matrix[:, latent_index]
    return assignment_frame


def main() -> None:
    config = ClusteringConfig()
    set_random_seed(config.RANDOM_SEED)

    output_paths = ensure_output_directories(stage_output_folder=configured_path(config.OUTPUT_FOLDER))
    clustering_root = output_paths["clustering_root"]
    preprocess_input_root = configured_path(config.PREPROCESS_INPUT_FOLDER)
    training_input_root = configured_path(config.TRAINING_INPUT_FOLDER)

    logger.info("전처리 입력 폴더: %s", preprocess_input_root)
    logger.info("학습 입력 폴더: %s", training_input_root)
    logger.info("클러스터링 출력 폴더: %s", clustering_root)

    beat_features, learning_input_columns, latent_frame, split_info, scaler, dae_checkpoint = load_clustering_inputs(
        preprocess_root=preprocess_input_root,
        representation_root=training_input_root,
    )
    validate_feature_inputs(
        beat_features=beat_features,
        learning_input_columns=learning_input_columns,
        latent_frame=latent_frame,
    )

    split_data = build_split_frames(beat_features, split_info)
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
    all_valid_matrix = dataframe_to_feature_matrix(beat_features, learning_input_columns)

    train_scaled = scaler.transform(train_matrix).astype(np.float32)
    val_scaled = scaler.transform(val_matrix).astype(np.float32)
    test_scaled = scaler.transform(test_matrix).astype(np.float32)
    all_valid_scaled = scaler.transform(all_valid_matrix).astype(np.float32)

    input_dim = train_scaled.shape[1]
    model, history_frame, best_centers = train_idec(
        train_matrix=train_scaled,
        val_matrix=val_scaled,
        dae_checkpoint=dae_checkpoint,
        input_dim=input_dim,
        config=config,
        clustering_root=clustering_root,
    )

    test_recon_loss, test_cluster_loss, test_total_loss = evaluate_idec(
        model=model,
        matrix=test_scaled,
        batch_size=config.BATCH_SIZE,
        cluster_loss_weight=config.CLUSTER_LOSS_WEIGHT,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    logger.info(
        "test_recon_loss=%.8f test_cluster_loss=%.8f test_total_loss=%.8f",
        test_recon_loss,
        test_cluster_loss,
        test_total_loss,
    )

    np.save(clustering_root / config.CLUSTER_CENTERS_FILENAME, best_centers)
    history_frame.to_csv(clustering_root / config.HISTORY_FILENAME, index=False)

    assignment_frame = infer_assignments(
        model=model,
        feature_frame=beat_features,
        matrix=all_valid_scaled,
        batch_size=config.BATCH_SIZE,
    )
    assignment_path = clustering_root / config.ASSIGNMENTS_FILENAME
    assignment_frame.to_csv(assignment_path, index=False)
    logger.info("cluster assignment export 경로: %s", assignment_path)


if __name__ == "__main__":
    main()
