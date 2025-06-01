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
 
# Thời gian chờ giữa các hành động
last_action_time = 0
action_delay = 1.5  # Giây
 
# Tạo file ghi log nếu chưa có
log_file = "hand_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Thời gian", "Tay", "Số ngón", "Hướng lòng bàn tay", "Hành động"])
 
# Đếm số ngón tay đang giơ lên
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
 
# Xác định lòng bàn tay có đang hướng về camera không
def is_palm_facing_camera(landmarks):
    point0 = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    point5 = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    point17 = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
 
    v1 = point5 - point0
    v2 = point17 - point0
    normal_vector = np.cross(v1, v2)
    normal_vector /= np.linalg.norm(normal_vector)
 
    return normal_vector[2] < 0  # Hướng ra phía trước camera
 
# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
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
            label = handedness.classification[0].label  # "Left" hoặc "Right"
            finger_count = count_fingers(landmarks, label)
            palm_facing = is_palm_facing_camera(landmarks)
            now = time.time()
 
            action = ""
            if now - last_action_time > action_delay:
                if finger_count == 5 and palm_facing:
                    pyautogui.press("f5")
                    action = "Bắt đầu trình chiếu"
                elif finger_count == 0:
                    pyautogui.press("esc")
                    action = "Thoát trình chiếu"
                elif finger_count == 1:
                    if label == "Left":
                        pyautogui.press("left")
                        action = "Lùi trang"
                    elif label == "Right":
                        pyautogui.press("right")
                        action = "Tiến trang"
 
                if action:
                    last_action_time = now
                    print(f"🖐 {label} - {finger_count} ngón - Hành động: {action}")
 
                    # Ghi log
                    with open(log_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            time.strftime('%Y-%m-%d %H:%M:%S'),
                            label,
                            finger_count,
                            palm_facing,
                            action
                        ])
 
            # Hiển thị thông tin trên khung hình
            cv2.putText(frame, f"{label} - {finger_count} ngón", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
 
    # Hướng dẫn sử dụng
    guide_y = frame.shape[0] - 160
    cv2.rectangle(frame, (10, guide_y - 20), (600, guide_y + 140), (50, 50, 50), -1)
 
    cv2.putText(frame, "📌 Hướng dẫn sử dụng:", (20, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "✋ Giơ 5 ngón tay (hướng về camera): Bắt đầu trình chiếu", (20, guide_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, "✊ Nắm tay: Thoát trình chiếu", (20, guide_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    cv2.putText(frame, "👈 Tay trái giơ 1 ngón: Lùi slide", (20, guide_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, "👉 Tay phải giơ 1 ngón: Tiến slide", (20, guide_y + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
 
    # Hiển thị cửa sổ chính
    cv2.imshow("🎥 Điều khiển PowerPoint bằng tay", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
