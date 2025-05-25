from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ObjectDetector:
    """A class for detecting objects in images using YOLOv8."""

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.25):
        """
        Initialize the ObjectDetector with a YOLO model.

        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections (0-1)
        """
        try:
            logger.info(f"Initializing YOLO model from {model_path}")
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Object detector initialized successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing object detector: {str(e)}")
            raise

    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess the input image for detection.

        Args:
            image: Input image (PIL Image, file path, or numpy array)

        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            if isinstance(image, (str, Path)):
                image = Image.open(image)

            if isinstance(image, Image.Image):
                image = np.array(image)

            if len(image.shape) == 2:  # Convert grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # Convert RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def detect(self, image: Union[str, Path, Image.Image, np.ndarray]) -> List[Dict]:
        """
        Detect objects in the given image.

        Args:
            image: Input image (PIL Image, file path, or numpy array)

        Returns:
            List of dictionaries containing detection results:
            [
                {
                    'name': str,         # Class name
                    'confidence': float, # Detection confidence
                    'box': List[float],  # Bounding box [x1, y1, x2, y2]
                    'class_id': int      # Class ID
                },
                ...
            ]
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(image)

            # Run inference
            logger.debug("Running YOLO inference")
            results = self.model(
                img_array,
                conf=self.conf_threshold,
                device=self.device
            )

            # Process detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        name = self.model.names[cls_id]
                        coords = box.xyxy[0].tolist()

                        detection = {
                            'name': name,
                            'confidence': conf,
                            'box': coords,
                            'class_id': cls_id
                        }
                        detections.append(detection)

                    except Exception as e:
                        logger.warning(f"Error processing detection: {str(e)}")
                        continue

            logger.info(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            return []

    def detect_and_draw(
            self,
            image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[List[Dict], Optional[Image.Image]]:
        """
        Detect objects and draw bounding boxes on the image with improved small object detection.
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(image)

            results = self.model(
                img_array,
                conf=0.25,
                imgsz=1280,
                augment=True
            )

            # Get detections with dynamic confidence threshold
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.model.names[cls_id]
                    coords = box.xyxy[0].tolist()

                    # Calculate object size (area)
                    width = coords[2] - coords[0]
                    height = coords[3] - coords[1]
                    area = width * height
                    img_area = img_array.shape[0] * img_array.shape[1]
                    relative_size = area / img_area

                    # Adjust confidence threshold based on object size
                    size_threshold = 0.5 if relative_size < 0.1 else self.conf_threshold

                    if conf >= size_threshold:
                        detection = {
                            'name': name,
                            'confidence': conf,
                            'box': coords,
                            'class_id': cls_id
                        }
                        detections.append(detection)

            # Create annotated image
            annotated_img = Image.fromarray(results[0].plot())

            return detections, annotated_img

        except Exception as e:
            logger.error(f"Error in detect_and_draw: {str(e)}")
            return [], None
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.model.type,
            'model_task': self.model.task,
            'num_classes': len(self.model.names),
            'class_names': self.model.names,
            'device': str(self.device),
            'confidence_threshold': self.conf_threshold
        }

    def cleanup(self):
        """Clean up resources if needed."""
        try:
            if hasattr(self, 'model'):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleaned up object detector resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __call__(
            self,
            image: Union[str, Path, Image.Image, np.ndarray]
    ) -> List[Dict]:
        """Allow using the class instance as a callable."""
        return self.detect(image)