
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceManager:
    def __init__(self):
        self.model = None
        logger.info(f"model 'trained_model.pt' is loaded.")

    def reload_model(self,):
        del self.model
        # Load a model
        self.model = YOLO("trained_model.pt")
        logger.info(f"model 'trained_model.pt' has reloaded.")

        return self.model

    def inference(self, img):
        if self.model is None:
            self.reload_model()
            return self.model(img)
        else:
            return self.model(img)

    def unload_model(self,):
        del self.model
        self.model = None
