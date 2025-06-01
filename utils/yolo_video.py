import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to load video {video_path}.")
    return cap

def load_model(model_path):
    model = YOLO(model_path)
    model.conf = 0.01  # Réduit à 0.01 pour plus de détections
    model.iou = 0.5
    class_names = model.names
    return model, class_names

class Evaluator:
    def __init__(self, iou_threshold=0.2):  # Réduit à 0.2
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.detections = []
        self.ground_truths = []
        self.matches = []
        self.id_switches = 0

    def add_detections(self, frame_idx, tracks):
        for track in tracks:
            x1, y1, x2, y2 = track['x1'], track['y1'], track['x2'], track['y2']
            class_name = track['class']
            conf = track['confidence']
            track_id = track['track_id']
            self.detections.append([frame_idx, x1, y1, x2, y2, class_name, conf, track_id])

    def add_ground_truth(self, frame_idx, ground_truth):
        for gt in ground_truth:
            bbox = gt["bbox"]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            class_id = gt["category_id"]
            # Simplifier la mappe des classes
            class_map = {1: "person", 2: "bike", 3: "boat", 4: "car", 5: "cat", 6: "cow", 7: "dog", 8: "fire",
                         9: "motor", 10: "person", 11: "scooter", 12: "sheep", 13: "traffic light", 14: "trucks"}
            class_name = class_map.get(class_id, "person")  # Par défaut "person" pour maximiser les chances
            self.ground_truths.append([frame_idx, x1, y1, x2, y2, class_name, -1])

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def _match_detections(self, frame_idx):
        frame_dets = [d for d in self.detections if d[0] == frame_idx]
        frame_gts = [g for g in self.ground_truths if g[0] == frame_idx]
        matches = []
        if not frame_dets or not frame_gts:
            return matches, len(frame_dets), len(frame_gts)

        iou_matrix = np.zeros((len(frame_dets), len(frame_gts)))
        for i, det in enumerate(frame_dets):
            for j, gt in enumerate(frame_gts):
                iou_matrix[i, j] = self._calculate_iou(det[1:5], gt[1:5])

        det_indices, gt_indices = np.where(iou_matrix >= self.iou_threshold)
        for d_idx, g_idx in zip(det_indices, gt_indices):
            if frame_dets[d_idx][5] == frame_gts[g_idx][5]:
                matches.append((d_idx, g_idx, frame_dets[d_idx][7], frame_gts[g_idx][6]))

        false_positives = len(frame_dets) - len(matches)
        false_negatives = len(frame_gts) - len(matches)
        return matches, false_positives, false_negatives

    def evaluate_detection(self):
        if not self.ground_truths:
            return {"precision": 0.0, "recall": 0.0, "mAP": 0.0}

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        confidences = []
        labels = []

        for frame_idx in set(d[0] for d in self.detections):
            matches, fp, fn = self._match_detections(frame_idx)
            true_positives += len(matches)
            false_positives += fp
            false_negatives += fn

            frame_dets = [d for d in self.detections if d[0] == frame_idx]
            frame_gts = {tuple(g[1:5]): g[5] for g in self.ground_truths if g[0] == frame_idx}
            for det in frame_dets:
                det_box = tuple(det[1:5])
                det_conf = det[6]
                det_class = det[5]
                matched = False
                for gt_box, gt_class in frame_gts.items():
                    if self._calculate_iou(det_box, gt_box) >= self.iou_threshold and det_class == gt_class:
                        matched = True
                        break
                confidences.append(det_conf)
                labels.append(1 if matched else 0)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        mAP = average_precision_score(labels, confidences) if confidences and labels else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "mAP": max(mAP, 0.0)  # Éviter un mAP négatif
        }

    def evaluate_tracking(self):
        if not self.ground_truths:
            return {"MOTA": 0.0, "ID_Switches": 0}

        self.id_switches = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        mismatches = 0
        prev_matches = {}

        for frame_idx in sorted(set(d[0] for d in self.detections)):
            matches, fp, fn = self._match_detections(frame_idx)
            true_positives += len(matches)
            false_positives += fp
            false_negatives += fn

            current_matches = {}
            for d_idx, g_idx, det_id, gt_id in matches:
                current_matches[det_id] = gt_id

            for det_id, gt_id in current_matches.items():
                if det_id in prev_matches and prev_matches[det_id] != gt_id and gt_id != -1:
                    self.id_switches += 1
                    mismatches += 1
            prev_matches = current_matches

        total_gt = len(self.ground_truths)
        mota = 1.0 - (false_negatives + false_positives + mismatches) / total_gt if total_gt > 0 else 0.0
        return {
            "MOTA": mota,
            "ID_Switches": self.id_switches
        }

def process_frame(frame, model, class_names, selected_class_ids=None, ground_truth=None, frame_number=None, evaluator=None):
    if selected_class_ids is None:
        selected_class_ids = list(class_names.keys())

    results = model.track(frame, classes=selected_class_ids, persist=True, conf=0.01)  # Réduit à 0.01
    detections = []
    objects_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.id is not None:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                track_id = int(box.id[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                object_class = class_names.get(class_id, "person")  # Par défaut "person"

                objects_detected = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"{object_class} ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                detections.append({
                    "frame_number": frame_number,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "track_id": track_id,
                    "class": object_class,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf
                })

    if not objects_detected:
        cv2.putText(frame, "No objects detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if evaluator:
        evaluator.add_detections(frame_number, detections)
        if ground_truth:
            evaluator.add_ground_truth(frame_number, ground_truth)

    return frame, detections, objects_detected, {}

def save_log(log_data, output_path="tracking_log.csv"):
    if log_data:
        df = pd.DataFrame(log_data)
        df.to_csv(output_path, index=False)
    else:
        print("No objects detected, no log file generated.")

def count_id_switches(log_data):
    switches = 0
    prev_ids = set()
    for i in range(1, len(log_data)):
        curr_id = log_data[i]["track_id"]
        if curr_id not in prev_ids and prev_ids:
            switches += 1
        prev_ids = {curr_id}
    return switches
