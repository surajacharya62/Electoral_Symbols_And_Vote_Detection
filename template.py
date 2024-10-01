import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s')

model_name = "detectron2"
model_name2 = "faster_rcnn"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{model_name}/__init__.py",
    f"src/{model_name}/components/__init__.py",
    f"src/{model_name}/utils/__init__.py",
    f"src/{model_name}/config/__init__.py",
    f"src/{model_name}/config/configuration.py",
    f"src/{model_name}/pipeline/__init__.py",
    f"src/{model_name}/entity/__init__.py",
    f"src/{model_name}/constants/__init__.py",

    f"src/{model_name2}/__init__.py",
    f"src/{model_name2}/components/__init__.py",
    f"src/{model_name2}/utils/__init__.py",
    f"src/{model_name2}/config/__init__.py",
    f"src/{model_name2}/config/configuration.py",
    f"src/{model_name2}/pipeline/__init__.py",
    f"src/{model_name2}/entity/__init__.py",
    f"src/{model_name2}/constants/__init__.py",

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

