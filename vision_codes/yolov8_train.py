from ultralytics import YOLO

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_yolov8_model(data_path, epochs=50):
    model = YOLO('yolov8s-cls.pt')  # Load a pretrained model
    model.train(data=data_path, epochs=epochs, patience=20, verbose=True, project='langlearn_trains/',
                val=True, save_dir='./train_results/')

    model.save('trained_model.pt')
    logger.info(f"Train is finished...")
