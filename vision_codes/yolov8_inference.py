import numpy as np
from ultralytics import YOLO
import logging
from ultralytics.utils.ops import scale_image
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceManager:
    def __init__(self):
        self.cls = False
        self.model = None
        self.results = None
        logger.info(f"model 'trained_model.pt' is loaded.")

    def reload_model(self, ):
        del self.model
        # Load a model
        self.model = YOLO("trained_model.pt")
        logger.info(f"model 'trained_model.pt' has reloaded.")

        return self.model

    def inference(self, img):
        if self.model is None:
            self.reload_model()
            return self.process_yolov8_results(self.model(img))
        else:
            return self.process_yolov8_results(self.model(img))

    def unload_model(self, ):
        del self.model
        self.model = None

    def process_yolov8_results(self, results, image_name="test.png"):
        """
        Processes the YOLOv8 results and prepares a JSON response.

        Args:
        results (ultralytics.engine.results.Results): The results object from YOLOv8 model.
        image_name (str): The name of the image.

        Returns:
        str: JSON formatted string containing the predictions.
        """

        response = []
        for i, result in enumerate(results):
            predictions = []
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk

            for box in result.boxes:
                class_idx = int(box.cls.item())
                bbox = box.xyxy[0].cpu().numpy().tolist()  # Extract the bbox as a list
                prediction = {
                    "class_name": result.names[class_idx],
                    "bbox": bbox,
                    "confidence": box.conf.item()
                }
                predictions.append(prediction)

            response.append({
                "image_name": i,
                "predictions": predictions
            })

        return json.dumps(response)