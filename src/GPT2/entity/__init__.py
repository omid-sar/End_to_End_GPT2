from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    dataset: str
    local_data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list 

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    dataset_name: str
    dataset: str
    downloaded_files: Path
    local_data_file: Path
    shard_size: int
    total_batch_size: int
    B: int
    T: int


@dataclass(frozen=True)
class GPTConfig:
    root_dir: Path
    verification_info_dir: Path
    verification_summary_file: Path
    verification_weights_file: Path
    block_size : int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd : int
    weight_decay: float
    learning_rate: float

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    model_folder: Path
    total_batch_size: int
    B: int
    T: int
    max_lr: float
    min_lr: float
    warmup_steps: int
    max_steps: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path

@dataclass(frozen=True)
class ModelInferenceConfig:
    root_dir: Path