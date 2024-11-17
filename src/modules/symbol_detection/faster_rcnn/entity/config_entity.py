## Configuration entity
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass
class EvaluationConfig:
    root_dir: Path
    path_of_model: Path
    test_images_path: Path
    annotations_path: Path
    faster_rcnn_files_path: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    classes: int