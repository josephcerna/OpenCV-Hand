import cv2
import mediapipe as mp
import serial
import time
import math

# Initialize serial connection to ESP32
# Replace 'COM3' with your ESP32's port (check in Arduino IDE)
try:
    ser = serial.Serial('COM3', 115200, timeout=1)
    time.sleep(2)  # Wait for connection to establish
    print("Connected to ESP32")
except:
    print("Error: Cannot connect to ESP32. Check COM port.")
    ser = None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Track only one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Hand tracking started. Press 'ESC' to exit.")

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points (p1-p2-p3)
    Returns angle at point p2 in degrees"""
    # Create vectors
    v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]
    v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z]
    
    # Calculate dot product and magnitudes
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in v1))
    magnitude2 = math.sqrt(sum(a ** 2 for a in v2))
    
    # Avoid division by zero
    if magnitude1 * magnitude2 == 0:
        return 0
    
    # Calculate angle in radians then convert to degrees
    angle_rad = math.acos(max(-1, min(1, dot_product / (magnitude1 * magnitude2))))
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def calculate_finger_curl(landmarks, finger_indices):
    """Calculate finger curl based on joint angles
    Returns servo angle (10-180) where 180 = fully closed, 10 = fully open"""
    
    # For fingers with 4 landmarks (index, middle, ring, pinky)
    if len(finger_indices) == 4:
        # Calculate angles at MCP and PIP joints
        mcp_angle = calculate_angle(
            landmarks[finger_indices[0]],  # Base (knuckle area)
            landmarks[finger_indices[1]],  # MCP joint
            landmarks[finger_indices[2]]   # PIP joint
        )
        
        pip_angle = calculate_angle(
            landmarks[finger_indices[1]],  # MCP joint
            landmarks[finger_indices[2]],  # PIP joint
            landmarks[finger_indices[3]]   # Fingertip
        )
        
        # Average the angles (both contribute to curl)
        avg_angle = (mcp_angle + pip_angle) / 2
        
    # For thumb (3 main joints)
    else:
        # Calculate angle at the main thumb joint
        avg_angle = calculate_angle(
            landmarks[finger_indices[0]],
            landmarks[finger_indices[1]],
            landmarks[finger_indices[2]] if len(finger_indices) > 2 else landmarks[finger_indices[1]]
        )
    
    # Map joint angle to servo angle
    # Open hand: ~180° joint -> 10° servo (open)
    # Closed hand: ~90° joint -> 180° servo (closed)
    # Formula: map 90-180 range to 180-10 range (inverted)
    
    servo_angle = int((180 - avg_angle) * 2)
    servo_angle = max(10, min(170, servo_angle))  # Constrain to 10-170 (stops servo overextension)
    
    return servo_angle

# Finger landmark indices
THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks with custom style
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),  # Landmarks (white dots)
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)  # Connections (black lines)
            )
            
            # Get image dimensions
            h, w, c = frame.shape
            
            # Display landmark numbers on each point
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                # Draw landmark number next to each point
                cv2.putText(frame, str(idx), (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Calculate angles for each finger
            landmarks = hand_landmarks.landmark
            thumb_angle = calculate_finger_curl(landmarks, THUMB)
            index_angle = calculate_finger_curl(landmarks, INDEX)
            middle_angle = calculate_finger_curl(landmarks, MIDDLE)
            ring_angle = calculate_finger_curl(landmarks, RING)
            pinky_angle = calculate_finger_curl(landmarks, PINKY)
            
            # Display angle on each fingertip
            # Thumb tip (landmark 4)
            thumb_tip = landmarks[4]
            cv2.putText(frame, str(thumb_angle), 
                       (int(thumb_tip.x * w), int(thumb_tip.y * h) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 102, 0), 2)
            
            # Index tip (landmark 8)
            index_tip = landmarks[8]
            cv2.putText(frame, str(index_angle), 
                       (int(index_tip.x * w), int(index_tip.y * h) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 102, 0), 2)
            
            # Middle tip (landmark 12)
            middle_tip = landmarks[12]
            cv2.putText(frame, str(middle_angle), 
                       (int(middle_tip.x * w), int(middle_tip.y * h) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 102, 0), 2)
            
            # Ring tip (landmark 16)
            ring_tip = landmarks[16]
            cv2.putText(frame, str(ring_angle), 
                       (int(ring_tip.x * w), int(ring_tip.y * h) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 102, 0), 2)
            
            # Pinky tip (landmark 20)
            pinky_tip = landmarks[20]
            cv2.putText(frame, str(pinky_angle), 
                       (int(pinky_tip.x * w), int(pinky_tip.y * h) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 102, 0), 2)
            
            # Send data to ESP32
            if ser and ser.is_open:
                command = f"T:{thumb_angle},I:{index_angle},M:{middle_angle},R:{ring_angle},P:{pinky_angle}\n"
                ser.write(command.encode())
            
            # Display angles on screen
            cv2.putText(frame, f"Thumb: {thumb_angle}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 102, 0), 2)
            cv2.putText(frame, f"Index: {index_angle}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 102, 0), 2)
            cv2.putText(frame, f"Middle: {middle_angle}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 102, 0), 2)
            cv2.putText(frame, f"Ring: {ring_angle}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 102, 0), 2)
            cv2.putText(frame, f"Pinky: {pinky_angle}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 102, 0), 2)
    
    cv2.imshow('Hand Tracking', frame)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
hands.close()
if ser:
    ser.close()