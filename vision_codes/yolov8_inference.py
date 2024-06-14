import io
import json

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from ultralytics.utils.ops import scale_image


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

    def process_yolov8_results(self, results):
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
            # result.show()  # display to screen
            # result.save(filename="result.jpg")  # save to disk
            for box in result.boxes:
                class_idx = int(box.cls.item())
                bbox = box.xyxy.cpu().numpy().astype(int).flatten()
                class_name = result.names[class_idx]
                confidence = box.conf.item()

                prediction = {
                    "class_name": class_name,
                    "bbox": bbox.tolist(),
                    "confidence": float("{:.2f}".format(confidence * 100))
                }
                predictions.append(prediction)


            response.append({
                "image_name": i,
                "predictions": predictions
            })

        return json.dumps(response)

    def inference_on_video(self, video_path):
        if self.model is None:
            self.reload_model()
            return self.process_yolov8_results(self.model(video_path, stream=True, save=True))
        else:
            return self.process_yolov8_results(self.model(video_path, stream=True, save=True))
