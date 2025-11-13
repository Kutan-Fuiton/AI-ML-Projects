# Hand Landmark Detection - Single File
# Install these first: pip install mediapipe opencv-python numpy

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmark names for reference
landmark_names = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

print("🖐️ Starting Hand Landmark Detection...")
print("Press 'q' to quit, 's' for screenshot")

# Initialize hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access webcam")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

screenshot_count = 0
frame_count = 0
start_time = cv2.getTickCount()

try:
    while True:
        # Read frame
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read frame")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand detection
        results = hands.process(frame_rgb)
        
        # Draw landmarks and get hand data
        annotated_frame = frame.copy()
        hand_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get hand information
                hand_label = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                # Extract landmark coordinates
                h, w = frame.shape[:2]
                landmarks = []
                for idx, lm in enumerate(hand_landmarks.landmark):
                    landmarks.append({
                        'id': idx,
                        'name': landmark_names[idx],
                        'x': lm.x * w,
                        'y': lm.y * h,
                        'z': lm.z
                    })
                
                hand_data.append({
                    'landmarks': landmarks,
                    'label': hand_label,
                    'confidence': confidence
                })
                
                # Draw hand label near wrist
                if landmarks:
                    wrist = landmarks[0]
                    info_text = f"{hand_label} Hand: {confidence:.2f}"
                    cv2.putText(annotated_frame, info_text, 
                               (int(wrist['x']) - 50, int(wrist['y']) - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Count fingers for each hand
        if hand_data:
            for i, hand in enumerate(hand_data):
                landmarks = hand['landmarks']
                finger_count = 0
                
                # Check each finger (thumb to pinky)
                finger_tips = [4, 8, 12, 16, 20]
                finger_pips = [2, 6, 10, 14, 18]
                
                for tip_idx, pip_idx in zip(finger_tips, finger_pips):
                    if tip_idx < len(landmarks) and pip_idx < len(landmarks):
                        if landmarks[tip_idx]['y'] < landmarks[pip_idx]['y']:
                            finger_count += 1
                
                # Display finger count
                cv2.putText(annotated_frame, f"Hand {i+1}: {finger_count} fingers",
                           (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate FPS
        frame_count += 1
        current_time = cv2.getTickCount()
        time_elapsed = (current_time - start_time) / cv2.getTickFrequency()
        fps = frame_count / time_elapsed
        
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                   (10, annotated_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display instructions
        cv2.putText(annotated_frame, "Press 'q' to quit | 's' for screenshot",
                   (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Hand Landmark Detection', annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            screenshot_count += 1
            filename = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Saved {filename}")
        elif key == ord('1'):
            # Show landmark numbers
            if hand_data:
                for hand in hand_data:
                    for landmark in hand['landmarks']:
                        x, y = int(landmark['x']), int(landmark['y'])
                        cv2.putText(annotated_frame, str(landmark['id']), 
                                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

except KeyboardInterrupt:
    print("\n🛑 Program interrupted by user")
except Exception as e:
    print(f"❌ Error: {e}")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"✅ Program ended. Processed {frame_count} frames")
    print(f"📸 Screenshots taken: {screenshot_count}")