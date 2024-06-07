import shutil

from ultralytics import YOLO
import os.path

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_yolov8_model(data_path, epochs=50):
    if os.path.exists('trained_model.pt'):
        logger.info(f"Train on ==> trained_model.pt")
        model = YOLO('trained_model.pt')  # Load a pretrained model
    else:
        model = YOLO('yolov8s-cls.pt')

    results = model.train(data=data_path, epochs=epochs, patience=20, verbose=True, project='langlearn_trains/',
                          val=True, save_dir='./train_results/')

    try:
        os.remove("trained_model.pt")
        logger.info(f"removed old model")
    except OSError:
        pass

    src = f"{results.save_dir}/weights/best.pt"
    dst = f"trained_model.pt"
    shutil.copyfile(src, dst)
    logger.info(f"Train is finished...")
