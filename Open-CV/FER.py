import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print("😊 Starting Facial Emotion Recognition...")
print("Press 'q' to quit, 's' for screenshot")

# Initialize face detection and mesh
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Emotion detection based on facial landmarks
def detect_emotion(landmarks, frame_shape):
    if landmarks is None:
        return "No Face"
    
    h, w = frame_shape[:2]
    
    # Get key landmarks
    try:
        # Mouth landmarks
        upper_lip = landmarks[13]  # Upper lip
        lower_lip = landmarks[14]  # Lower lip
        mouth_left = landmarks[61]  # Mouth left corner
        mouth_right = landmarks[291]  # Mouth right corner
        
        # Eye landmarks
        left_eye_inner = landmarks[133]  # Left eye inner corner
        left_eye_outer = landmarks[33]   # Left eye outer corner
        right_eye_inner = landmarks[362] # Right eye inner corner
        right_eye_outer = landmarks[263] # Right eye outer corner
        
        # Eyebrow landmarks
        left_eyebrow = landmarks[105]    # Left eyebrow
        right_eyebrow = landmarks[334]   # Right eyebrow
        
        # Calculate features
        mouth_height = abs(upper_lip.y - lower_lip.y) * h
        mouth_width = abs(mouth_left.x - mouth_right.x) * w
        
        left_eye_openness = abs(left_eye_inner.y - left_eye_outer.y) * h
        right_eye_openness = abs(right_eye_inner.y - right_eye_outer.y) * h
        
        eyebrow_height = ((left_eyebrow.y + right_eyebrow.y) / 2) * h
        
        # Emotion detection logic
        if mouth_height > 15 and mouth_width > 50:  # Big open mouth
            if left_eye_openness > 10 and right_eye_openness > 10:  # Eyes open
                return "Surprised 😲"
            else:
                return "Yawning 😴"
        
        elif mouth_width > 60 and mouth_height < 10:  # Wide but closed mouth
            return "Happy 😊"
        
        elif mouth_height < 5 and mouth_width < 40:  # Small closed mouth
            if eyebrow_height < 200:  # Lowered eyebrows
                return "Angry 😠"
            else:
                return "Neutral 😐"
        
        elif left_eye_openness < 8 or right_eye_openness < 8:  # Squinting eyes
            return "Suspicious 🤨"
        
        else:
            return "Neutral 😐"
            
    except Exception as e:
        return "Analyzing 🔍"

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

# Colors for different emotions
emotion_colors = {
    "Happy 😊": (0, 255, 0),      # Green
    "Surprised 😲": (255, 255, 0), # Yellow
    "Angry 😠": (0, 0, 255),      # Red
    "Neutral 😐": (255, 255, 255), # White
    "Suspicious 🤨": (255, 165, 0), # Orange
    "Yawning 😴": (128, 0, 128),   # Purple
    "No Face": (100, 100, 100),   # Gray
    "Analyzing 🔍": (0, 191, 255) # Deep Sky Blue
}

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
        
        # Process the frame for face detection
        face_results = face_detection.process(frame_rgb)
        mesh_results = face_mesh.process(frame_rgb)
        
        annotated_frame = frame.copy()
        emotion = "No Face"
        face_bbox = None
        
        # Draw face detection results
        if face_results.detections:
            for detection in face_results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                face_bbox = (x, y, width, height)
                
                # Draw face bounding box
                cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Detect emotion from face mesh
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Draw face mesh
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Detect emotion
                emotion = detect_emotion(face_landmarks.landmark, frame.shape)
        
        # Calculate FPS
        frame_count += 1
        current_time = cv2.getTickCount()
        time_elapsed = (current_time - start_time) / cv2.getTickFrequency()
        fps = frame_count / time_elapsed
        
        # Get color for current emotion
        emotion_color = emotion_colors.get(emotion, (255, 255, 255))
        
        # Display emotion with colored background
        text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = 10
        text_y = 50
        
        # Draw background rectangle for emotion text
        cv2.rectangle(annotated_frame, (text_x - 5, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 10, text_y + 5), emotion_color, -1)
        
        # Draw emotion text
        cv2.putText(annotated_frame, emotion, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)  # Black text
        
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                   (10, annotated_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display instructions
        cv2.putText(annotated_frame, "Press 'q' to quit | 's' for screenshot",
                   (10, annotated_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Facial Emotion Recognition', annotated_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            screenshot_count += 1
            filename = f"emotion_screenshot_{screenshot_count}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Saved {filename}")
        elif key == ord('1'):
            # Toggle face mesh
            pass

except KeyboardInterrupt:
    print("\n🛑 Program interrupted by user")
except Exception as e:
    print(f"❌ Error: {e}")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()
    face_mesh.close()
    print(f"✅ Program ended. Processed {frame_count} frames")
    print(f"📸 Screenshots taken: {screenshot_count}")