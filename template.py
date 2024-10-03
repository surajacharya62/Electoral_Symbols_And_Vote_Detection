import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s')

model_name1 = "detectron2"
model_name2 = "faster_rcnn"
model_name3 = "yolov8"

list_of_files = [
    ".github/workflows/.gitkeep",
    "docs/STRUCTURE.md",
    f"src/modules/ballot_creation/__init__.py",
    f"src/modules/symbol_detection/{model_name1}/__init__.py",
    f"src/modules/symbol_detection/{model_name1}/components/__init__.py",
    f"src/modules/symbol_detection/{model_name1}/utils/__init__.py",
    f"src/modules/symbol_detection/{model_name1}/config/__init__.py",
    f"src/modules/symbol_detection/{model_name1}/config/configuration.py",
    f"src/modules/symbol_detection/{model_name1}/pipeline/__init__.py",
    f"src/modules/symbol_detection/{model_name1}/entity/__init__.py",
    f"src/modules/symbol_detection/{model_name1}/constants/__init__.py",
    f"src/modules/vote_validation/{model_name1}/__init__.py",
    
    f"src/modules/symbol_detection/{model_name2}/__init__.py",
    f"src/modules/symbol_detection/{model_name2}/components/__init__.py",
    f"src/modules/symbol_detection/{model_name2}/utils/__init__.py",
    f"src/modules/symbol_detection/{model_name2}/config/__init__.py",
    f"src/modules/symbol_detection/{model_name2}/config/configuration.py",
    f"src/modules/symbol_detection/{model_name2}/pipeline/__init__.py",
    f"src/modules/symbol_detection/{model_name2}/entity/__init__.py",
    f"src/modules/symbol_detection/{model_name2}/constants/__init__.py",
    f"src/modules/vote_validation/{model_name2}/__init__.py",
  
    f"src/modules/symbol_detection/{model_name3}/__init__.py",
    f"src/modules/symbol_detection/{model_name3}/components/__init__.py",
    f"src/modules/symbol_detection/{model_name3}/utils/__init__.py",
    f"src/modules/symbol_detection/{model_name3}/config/__init__.py",
    f"src/modules/symbol_detection/{model_name3}/config/configuration.py",
    f"src/modules/symbol_detection/{model_name3}/pipeline/__init__.py",
    f"src/modules/symbol_detection/{model_name3}/entity/__init__.py",
    f"src/modules/symbol_detection/{model_name3}/constants/__init__.py",
    f"src/modules/vote_validation/{model_name3}/__init__.py",

    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/detectron2_training.ipynb",
    "templates/index.html"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")

