# main.py
import cv2
import time
import json
import os
from typing import Dict, List, Tuple
import numpy as np
from src.detector import ObjectDetector, SubObjectDetector
from src.config import OUTPUT_DIR, SUBOBJECT_IMAGES_DIR


class ObjectDetectionSystem:
    def __init__(self):
        print("Initializing Object Detection System...")
        self.object_detector = ObjectDetector(conf_thresh=0.3)  # Lowered threshold
        self.sub_object_detector = SubObjectDetector()
        self.detections = []

        # Ensure output directories exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(SUBOBJECT_IMAGES_DIR, exist_ok=True)
        print("Initialization complete.")

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        # Detect main objects
        detections = self.object_detector.detect(frame)
        frame_detections = []

        if detections:
            print(f"Found {len(detections)} main objects in frame")

        # Process each detection
        for detection in detections:
            print(f"Processing {detection['object']} with confidence {detection['confidence']:.2f}")

            # Detect sub-objects
            sub_object = self.sub_object_detector.detect(frame, detection)

            if sub_object:
                # Format detection
                result = {
                    "object": detection["object"],
                    "id": detection["id"],
                    "bbox": [round(x, 2) for x in detection["bbox"]],
                    "subobject": {
                        "object": sub_object["object"],
                        "id": sub_object["id"],
                        "bbox": [round(x, 2) for x in sub_object["bbox"]]
                    }
                }
                frame_detections.append(result)
                print(f"Found {sub_object['object']} for {detection['object']} {detection['id']}")

                # Save sub-object image
                self.save_subobject_image(frame, result)

                # Draw detections
                self.draw_detection(frame, result)

        return frame_detections, frame

    def save_subobject_image(self, frame: np.ndarray, detection: Dict):
        try:
            sub_obj = detection["subobject"]
            x1, y1, x2, y2 = map(int, sub_obj["bbox"])

            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                sub_obj_img = frame[y1:y2, x1:x2]
                if sub_obj_img.size > 0:
                    filename = f"{detection['object']}_{detection['id']}_{sub_obj['object']}.jpg"
                    filepath = os.path.join(SUBOBJECT_IMAGES_DIR, filename)
                    cv2.imwrite(filepath, sub_obj_img)
                    print(f"Saved sub-object image: {filename}")
        except Exception as e:
            print(f"Error saving sub-object image: {e}")

    def draw_detection(self, frame: np.ndarray, detection: Dict):
        try:
            # Draw main object
            x1, y1, x2, y2 = map(int, detection["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection['object']} {detection['id']}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw sub-object
            sub_obj = detection["subobject"]
            sx1, sy1, sx2, sy2 = map(int, sub_obj["bbox"])
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)
            cv2.putText(frame, sub_obj["object"], (sx1, sy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        except Exception as e:
            print(f"Error drawing detection: {e}")

    def process_video(self, video_path: str):
        print(f"\nProcessing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video properties: {total_frames} frames at {fps} FPS")

        frame_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every 2nd frame for speed
                if frame_count % 2 == 0:
                    # Resize frame for faster processing
                    frame = cv2.resize(frame, (640, 480))

                    # Process frame
                    frame_detections, processed_frame = self.process_frame(frame)

                    # Save detections
                    if frame_detections:
                        self.detections.extend(frame_detections)
                        print(f"\nFrame {frame_count} detections:")
                        print(json.dumps(frame_detections, indent=2))

                    # Display frame
                    cv2.imshow('Detection', processed_frame)

                    # Show FPS
                    if frame_count % 10 == 0:
                        elapsed_time = time.time() - start_time
                        current_fps = frame_count / elapsed_time
                        print(f"Progress: {frame_count}/{total_frames} frames ({current_fps:.2f} FPS)")

                frame_count += 1

                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing interrupted by user")
                    break

        except Exception as e:
            print(f"Error during video processing: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Save all detections
            if self.detections:
                output_file = os.path.join(OUTPUT_DIR, 'detections.json')
                with open(output_file, 'w') as f:
                    json.dump(self.detections, f, indent=2)
                print(f"\nSaved {len(self.detections)} detections to {output_file}")

            elapsed_time = time.time() - start_time
            final_fps = frame_count / elapsed_time

            print("\nProcessing Summary:")
            print(f"Total frames processed: {frame_count}")
            print(f"Processing time: {elapsed_time:.2f} seconds")
            print(f"Average FPS: {final_fps:.2f}")
            print(f"Total detections: {len(self.detections)}")
            print(f"Sub-object images saved: {len(os.listdir(SUBOBJECT_IMAGES_DIR))}")


def main():
    try:
        # Video path - update this to your video path
        video_path = r"C:\Users\udits\Downloads\1860079-uhd_2560_1440_25fps.mp4" # update with your video file path here

        # Create and run detection system
        system = ObjectDetectionSystem()
        system.process_video(video_path)

    except Exception as e:
        print(f"Error in main: {e}")

    finally:
        cv2.destroyAllWindows()
        print("\nProgram finished.")


if __name__ == "__main__":
    main()
