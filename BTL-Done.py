import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import csv
import os
 
# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)
drawing_utils = mp.solutions.drawing_utils
last_action_time = 0
action_delay = 1.5
 
# File ghi log
log_file = "hand_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "hand", "finger_count", "palm_facing", "action"])
 
# Hàm đếm số ngón tay
def count_fingers(landmarks, handedness):
    fingers = []
    if handedness == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return sum(fingers)
 
# Xác định lòng bàn tay
def is_palm_facing_camera(landmarks):
    point0 = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    point5 = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    point17 = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    v1 = point5 - point0
    v2 = point17 - point0
    normal_vector = np.cross(v1, v2)
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector[2] < 0
 
# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Tăng độ rộng khung hình
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Tăng độ cao khung hình

if not cap.isOpened():
    print("❌ Không thể mở webcam.")
    exit()

 
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
 
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            label = handedness.classification[0].label
            finger_count = count_fingers(landmarks, label)
            palm_facing = is_palm_facing_camera(landmarks)
            now = time.time()
 
            action = ""
            if now - last_action_time > action_delay:
                if finger_count == 5 and palm_facing:
                    pyautogui.press("f5")
                    action = "start"
                elif finger_count == 0:
                    pyautogui.press("esc")
                    action = "exit"
                elif finger_count == 1:
                    if label == "Left":
                        pyautogui.press("left")
                        action = "prev"
                    elif label == "Right":
                        pyautogui.press("right")
                        action = "next"
                if action:
                    last_action_time = now
                    print(f"🖐 {label} - {finger_count} fingers - Action: {action}")
 
                    # Ghi log
                    with open(log_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), label, finger_count, palm_facing, action])
 
            # Hiển thị thông tin
            cv2.putText(frame, f"{label} - {finger_count} fingers", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
 
    # Hướng dẫn
    guide_height = 120
    guide_y = frame.shape[0] - guide_height - 20

    cv2.rectangle(frame, (10, guide_y - 30), (420, guide_y + 100), (40, 40, 40), -1)
    cv2.putText(frame, "Gio 5 ngon tay: Bat dau", (20, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Nam tay: Thoat", (20, guide_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Tay trai gio 1 ngon: Slide truoc", (20, guide_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, "Tay phai gio 1 ngon: Slide tiep theo", (20, guide_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Dieu khien PowerPoint", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()