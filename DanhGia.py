import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

# Đọc file log
try:
    df = pd.read_csv("hand_log.csv")
except FileNotFoundError:
    print("❌ Không tìm thấy file 'hand_log.csv'. Hãy đảm bảo file nằm cùng thư mục với mã nguồn.")
    sys.exit()

# Kiểm tra xem có cột true_label không
if "true_label" not in df.columns:
    print("📌 File 'hand_log.csv' chưa có cột 'true_label'.")
    print("➡️ Vui lòng thêm nhãn đúng vào file để tính độ chính xác.")
    sys.exit()

# Tính độ chính xác
accuracy = (df["action"] == df["true_label"]).mean()
print("\n📊 Số lần mỗi hành động dự đoán:")
print(df["action"].value_counts())
print("\n✅ Số lần mỗi hành động thật (true_label):")
print(df["true_label"].value_counts())
print(f"\n🎯 Độ chính xác của mô hình: {accuracy * 100:.2f}%")

# Chuẩn hóa nhãn
pred_counts = df["action"].value_counts().sort_index()
true_counts = df["true_label"].value_counts().sort_index()
all_labels = sorted(set(pred_counts.index).union(set(true_counts.index)))

# Confusion matrix
cm = confusion_matrix(df["true_label"], df["action"], labels=all_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix for SVC")
plt.tight_layout()
plt.show()

# F1-score
f1 = f1_score(df["true_label"], df["action"], labels=all_labels, average="macro")
print(f"🎯 F1-Score trung bình: {f1*100:.2f}%")

# Tạo confusion matrix
cm = confusion_matrix(df["true_label"], df["action"], labels=all_labels)

# Thiết lập kích thước
plt.figure(figsize=(10, 8))

# Vẽ heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels, cbar=True)

# Vẽ biểu đồ số lần
plt.figure(figsize=(10, 6))
x = range(len(all_labels))
pred_values = [pred_counts.get(label, 0) for label in all_labels]
true_values = [true_counts.get(label, 0) for label in all_labels]
bar_width = 0.35
plt.bar(x, true_values, width=bar_width, label="True Labels", color="green")
plt.bar([i + bar_width for i in x], pred_values, width=bar_width, label="Predicted Actions", color="orange")
plt.xlabel("Hành động")
plt.ylabel("Số lần")
plt.title(f"🎯 Độ chính xác: {accuracy * 100:.2f}%")
plt.xticks([i + bar_width / 2 for i in x], all_labels)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
