import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Canvas to draw on
canvas = None
drawing = False  # Whether we're currently drawing
locked_point = None  # The point we're locked onto
current_color = (0, 255, 255)  # Yellow by default
thickness = 5

print("=== PRECISION AIR DRAWING ===")
print("INSTRUCTIONS:")
print("1. Show your finger to the camera")
print("2. Press 'l' to LOCK onto your fingertip")
print("3. Move your hand - the locked point will follow")
print("4. Press 'd' to START drawing with the locked point")
print("5. Press 'd' again to STOP drawing")
print("6. Press 'c' to clear the canvas")
print("7. Press '1', '2', '3' to change colors")
print("8. Press '+' and '-' to change thickness")
print("9. Press 's' to save your drawing")
print("10. Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Initialize canvas if not done yet
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Remove noise
    kernel = np.ones((5,5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_fingertip = None
    
    # Find the largest contour (likely the hand/finger)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Only proceed if the contour is reasonably large (not noise)
        if cv2.contourArea(largest_contour) > 1000:
            # Get the convex hull of the contour
            hull = cv2.convexHull(largest_contour)
            
            # Find the topmost point of the convex hull (likely the fingertip)
            topmost = tuple(hull[hull[:,:,1].argmin()][0])
            current_fingertip = topmost
            
            # Draw the convex hull (for visualization)
            cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
            
            # Draw a circle at the current fingertip position
            cv2.circle(frame, topmost, 8, (0, 255, 0), 2)  # Green circle for current position
    
    # If we have a locked point, make it follow the current fingertip
    if locked_point is not None and current_fingertip is not None:
        # Smooth movement: move locked point towards current fingertip
        # This creates a "following" effect instead of jumping directly
        follow_speed = 0.3  # Adjust this value (0.0 to 1.0) for smoother/faster following
        new_x = int(locked_point[0] + (current_fingertip[0] - locked_point[0]) * follow_speed)
        new_y = int(locked_point[1] + (current_fingertip[1] - locked_point[1]) * follow_speed)
        locked_point = (new_x, new_y)
    
    # Draw the locked point (if it exists)
    if locked_point is not None:
        # Draw a solid circle for the locked point
        color = (0, 0, 255) if drawing else (255, 0, 0)  # Red if drawing, Blue if not
        cv2.circle(frame, locked_point, 10, color, -1)
        
        # Draw from previous locked point to current locked point if drawing
        if drawing and prev_locked_point is not None:
            cv2.line(canvas, prev_locked_point, locked_point, current_color, thickness)
    
    # Store the current locked point for the next frame
    prev_locked_point = locked_point
    
    # Combine the canvas with the current frame
    result = frame.copy()
    mask = canvas.astype(bool)
    result[mask] = canvas[mask]
    
    # # Display status information
    # status_text = []
    # status_text.append(f"Locked: {'Yes' if locked_point else 'No'}")
    # status_text.append(f"Drawing: {'Yes' if drawing else 'No'}")
    # status_text.append(f"Color: {current_color}")
    # status_text.append(f"Thickness: {thickness}")
    
    # for i, text in enumerate(status_text):
    #     cv2.putText(result, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # # Display instructions
    # instructions = [
    #     "l: Lock point", "d: Toggle draw", "c: Clear", 
    #     "s: Save", "1/2/3: Colors", "+/-: Thickness", "q: Quit"
    # ]
    
    # for i, text in enumerate(instructions):
    #     cv2.putText(result, text, (result.shape[1] - 250, 30 + i*25), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Display the result
    cv2.imshow('Precision Air Drawing', result)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('l'):  # Lock onto current fingertip
        if current_fingertip is not None:
            locked_point = current_fingertip
            print(f"Locked onto point: {locked_point}")
        else:
            print("No fingertip detected to lock onto!")
    elif key == ord('d'):  # Toggle drawing
        drawing = not drawing
        print(f"Drawing: {'STARTED' if drawing else 'STOPPED'}")
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros_like(frame)
        print("Canvas cleared!")
    elif key == ord('s'):  # Save drawing
        cv2.imwrite('air_drawing_saved.jpg', canvas)
        print("Drawing saved as 'air_drawing_saved.jpg'")
    elif key == ord('1'):
        current_color = (0, 255, 255)  # Yellow
        print("Color changed to Yellow")
    elif key == ord('2'):
        current_color = (255, 0, 0)  # Blue
        print("Color changed to Blue")
    elif key == ord('3'):
        current_color = (0, 255, 0)  # Green
        print("Color changed to Green")
    elif key == ord('+'):
        thickness = min(thickness + 1, 20)
        print(f"Thickness increased to {thickness}")
    elif key == ord('-'):
        thickness = max(thickness - 1, 1)
        print(f"Thickness decreased to {thickness}")
    elif key == ord('u'):  # Unlock point
        locked_point = None
        drawing = False
        print("Point unlocked and drawing stopped")

# Release resources
cap.release()
cv2.destroyAllWindows()