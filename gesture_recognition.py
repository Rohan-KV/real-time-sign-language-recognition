"""
Hand Gesture Recognition for Sign Language
Uses MediaPipe Tasks Vision for hand tracking and OpenCV for image processing.
"""

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions, vision


class GestureRecognizer:
    """Captures hand gestures and identifies signs."""

    def __init__(self):
        # Create HandLandmarker options
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path='hand_landmarker.task'
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

    def count_fingers(self, landmarks):
        """Count extended fingers based on landmark positions."""
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        finger_count = 0

        # Thumb (check if extended)
        thumb_extended = abs(landmarks[4].x - landmarks[0].x) > abs(landmarks[3].x - landmarks[0].x)
        if thumb_extended:
            finger_count += 1

        # Other fingers (check if tip is above pip joint)
        for i in range(1, 5):
            if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
                finger_count += 1

        return finger_count

    def identify_gesture(self, landmarks, hand_label):
        """Identify the gesture based on finger positions."""
        finger_count = self.count_fingers(landmarks)

        # Check individual finger positions
        thumb_up = landmarks[4].y < landmarks[3].y
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y

        # Fist (all fingers down)
        if finger_count == 0:
            return "Fist"

        # Open Palm (all fingers up)
        if finger_count == 5:
            return "Open Palm"

        # Thumbs Up
        if thumb_up and finger_count == 1:
            return "Thumbs Up"

        # Thumbs Down
        if not thumb_up and finger_count == 1:
            return "Thumbs Down"

        # Peace/Victory Sign
        if index_up and middle_up and not ring_up and not pinky_up:
            return "Peace/Victory"

        # OK Sign
        if thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
            thumb_index_dist = np.sqrt(
                (landmarks[4].x - landmarks[8].x) ** 2 +
                (landmarks[4].y - landmarks[8].y) ** 2
            )
            if thumb_index_dist < 0.05:
                return "OK Sign"

        # Pointing
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "Pointing"

        # Rock On
        if index_up and pinky_up and not middle_up and not ring_up:
            return "Rock On"

        # Spider Man
        if index_up and middle_up and ring_up and pinky_up and not thumb_up:
            return "Spider Man"

        # Number gestures
        number_gestures = {
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five"
        }

        return number_gestures.get(finger_count, f"{finger_count} Fingers")

    def draw_landmarks(self, frame, landmarks, hand_label):
        """Draw hand landmarks on the frame."""
        h, w, _ = frame.shape

        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

        for i, j in connections:
            pt1 = (int(landmarks[i].x * w), int(landmarks[i].y * h))
            pt2 = (int(landmarks[j].x * w), int(landmarks[j].y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    def process_frame(self, frame):
        """Process a frame and detect gestures."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        result = self.hand_landmarker.detect(mp_image)

        gesture_info = []

        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                hand_label = result.handedness[i][0].display_name
                gesture = self.identify_gesture(landmarks, hand_label)

                h, w, _ = frame.shape
                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]
                x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

                gesture_info.append({
                    'gesture': gesture,
                    'hand_label': hand_label,
                    'bbox': (x_min, y_min, x_max, y_max),
                    'landmarks': landmarks
                })

                # Draw landmarks
                self.draw_landmarks(frame, landmarks, hand_label)

        return frame, gesture_info

    def run(self):
        """Run the gesture recognition from webcam."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Hand Gesture Recognition Started")
        print("Press 'q' to quit, 's' to save screenshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)

            # Process frame
            processed_frame, gestures = self.process_frame(frame)

            # Display gesture info
            y_offset = 30
            for info in gestures:
                text = f"{info['hand_label']} Hand: {info['gesture']}"
                cv2.putText(
                    processed_frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                y_offset += 40

            # Display instructions
            cv2.putText(
                processed_frame, "Press 'q' to quit, 's' to save", (10, processed_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

            cv2.imshow('Hand Gesture Recognition', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('gesture_screenshot.png', processed_frame)
                print("Screenshot saved as 'gesture_screenshot.png'")

        cap.release()
        cv2.destroyAllWindows()

    def cleanup(self):
        """Release resources."""
        self.hand_landmarker.close()


if __name__ == "__main__":
    recognizer = GestureRecognizer()
    try:
        recognizer.run()
    finally:
        recognizer.cleanup()
