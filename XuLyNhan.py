import os
import pandas as pd

# Hàm suy luận nhãn hành động từ số ngón và tay
def get_action_label(finger_count, hand):
    if finger_count == 5:
        return "start"
    elif finger_count == 0:
        return "exit"
    elif finger_count == 1:
        return "prev" if hand == "Left" else "next"
    else:
        return "none"

# Thêm cột true_label vào hand_log.csv nếu chưa có
def add_true_labels_to_log(log_path):
    if not os.path.exists(log_path):
        print(f"❌ Không tìm thấy file '{log_path}'.")
        return
    
    df = pd.read_csv(log_path)
    
    if "true_label" in df.columns:
        print("ℹ️ Cột 'true_label' đã tồn tại trong file log.")
    else:
        # Tính toán nhãn đúng từ dữ liệu log
        df["true_label"] = df.apply(
            lambda row: get_action_label(row["finger_count"], row["hand"]), axis=1
        )
        df.to_csv(log_path, index=False)
        print("✅ Đã thêm cột 'true_label' vào hand_log.csv.")

# Hàm tạo true labels từ thư mục dữ liệu huấn luyện/test
def extract_labels(data_dir):
    data = []
    for label_dir in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label_dir)
        if not os.path.isdir(label_path): continue
        for hand in ["left", "right"]:
            hand_path = os.path.join(label_path, hand)
            if not os.path.exists(hand_path): continue
            for filename in os.listdir(hand_path):
                if filename.endswith(".png"):
                    try:
                        parts = filename.split("_")[-1].replace(".png", "")
                        fingers = int(parts[0])
                        handed = 'Left' if parts[1].upper() == 'L' else 'Right'
                        true_label = get_action_label(fingers, handed)
                        data.append({
                            "filename": filename,
                            "finger_count": fingers,
                            "hand": handed,
                            "true_label": true_label
                        })
                    except Exception as e:
                        print(f"Lỗi với file {filename}: {e}")
    return pd.DataFrame(data)

# Đường dẫn file log
log_file = "hand_log.csv"
add_true_labels_to_log(log_file)

# Xử lý dữ liệu train/test nếu có
train_df = extract_labels(r"C:\Users\84377\Downloads\archive\fingers\train")
test_df = extract_labels(r"C:\Users\84377\Downloads\archive\fingers\test")

# Kết hợp và lưu
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df.to_csv("true_labels_from_dataset.csv", index=False)
print("✅ Đã lưu 'true_labels_from_dataset.csv' với nhãn đúng từ thư mục dữ liệu.")
