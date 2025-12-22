import cv2
from ultralytics import YOLO
import argparse
import os
from pathlib import Path


class HelmetDetector:
    def __init__(self, model_path='models/Helmet_Detection.pt', conf_threshold=0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        # Load the model
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect_image(self, image_path, save_path=None, show_result=True):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return

        # Run inference
        results = self.model(image, conf=self.conf_threshold)

        # Draw results on image
        annotated_image = results[0].plot()

        # Display results
        if show_result:
            cv2.imshow('Helmet Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save results
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            print(f"Result saved to {save_path}")

        # Print detection summary
        self.print_detections(results[0])

        return annotated_image

    def detect_video(self, video_path, save_path=None, show_result=True):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer if save path is provided
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')

            # Run inference
            results = self.model(frame, conf=self.conf_threshold)

            # Draw results on frame
            annotated_frame = results[0].plot()

            # Save frame if writer is initialized
            if writer:
                writer.write(annotated_frame)

            # Display frame
            if show_result:
                cv2.imshow('Helmet Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\nVideo processing complete!")
        if save_path:
            print(f"Output saved to {save_path}")

    def detect_webcam(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Starting webcam detection. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = self.model(frame, conf=self.conf_threshold)

            # Draw results on frame
            annotated_frame = results[0].plot()

            # Display frame
            cv2.imshow('Helmet Detection - Press q to quit', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def print_detections(self, result):
        """Print detection summary"""
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"Detected {len(boxes)} objects:")
            for i, box in enumerate(boxes):
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = self.model.names[cls]
                print(f"  {i + 1}. {class_name}: {conf:.2f}")
        else:
            print("No objects detected")


def main():
    parser = argparse.ArgumentParser(description='Run YOLO Helmet Detection')
    parser.add_argument('--model', default='models/helmv2.pt', help='Path to model file')
    parser.add_argument('--source', required=True, help='Source: image/video path or "webcam"')
    parser.add_argument('--output', help='Output path for saving results')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display results')

    args = parser.parse_args()

    # Initialize detector
    detector = HelmetDetector(args.model, args.conf)

    # Determine source type and run detection
    if args.source.lower() == 'webcam':
        detector.detect_webcam()
    elif os.path.isfile(args.source):
        # Check if it's an image or video
        ext = Path(args.source).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            detector.detect_image(args.source, args.output, not args.no_show)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            detector.detect_video(args.source, args.output, not args.no_show)
        else:
            print(f"Unsupported file format: {ext}")
    else:
        print(f"Source not found: {args.source}")


if __name__ == "__main__":
    # For Webcam as Input
    detector = HelmetDetector('../models/Helmet_Detection.pt', conf_threshold=0.5)
    detector.detect_webcam()

    # For Image as Input
    # detector.detect_image(f'../images/input/test.jpg', f'../images/output/result.jpg')

    # For Video as Input
    # detector.detect_video('../testvideo.mp4','result.mp4')
