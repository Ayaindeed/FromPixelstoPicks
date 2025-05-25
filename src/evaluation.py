from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self):
        """Initialize the evaluator"""
        logger.info("ModelEvaluator initialized successfully")

    def evaluate_object_detection(
            self,
            predictions: List[Dict],
            ground_truth: List[Dict],
            iou_threshold: float = 0.5
    ) -> Dict:
        """
        Calculate precision, recall, and F1-score for object detection.

        Args:
            predictions: List of predicted object dictionaries
            ground_truth: List of ground truth object dictionaries
            iou_threshold: IoU threshold for considering a detection as correct

        Returns:
            Dict containing precision, recall, and F1-score
        """
        try:
            true_positives = 0
            false_positives = 0
            false_negatives = len(ground_truth)

            for pred in predictions:
                pred_box = np.array(pred['box'])
                best_iou = 0
                best_gt_idx = -1

                for i, gt in enumerate(ground_truth):
                    gt_box = np.array(gt['box'])
                    iou = self._calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                # Check if detection is correct
                if best_iou >= iou_threshold:
                    if pred['name'] == ground_truth[best_gt_idx]['name']:
                        true_positives += 1
                        false_negatives -= 1
                else:
                    false_positives += 1

            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }

        except Exception as e:
            logger.error(f"Error evaluating object detection: {str(e)}")
            return {}

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0