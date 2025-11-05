import os
from pathlib import Path
import torch
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datetime import datetime

# ======================
# 🔧 Cấu hình cơ bản
# ======================
# Ép thư mục làm việc về gốc dự án
os.chdir(r"D:\DaihocDaiNam\Nam4\Ki1\Chuyen_doi_so\Baitaplon\exam_monitoring_vlm")
print("📂 Đã chuyển thư mục làm việc về:", os.getcwd())

# Đường dẫn model YOLO tuyệt đối
weights_path = Path(
    r"D:\DaihocDaiNam\Nam4\Ki1\Chuyen_doi_so\Baitaplon\exam_monitoring_vlm\runs\detect\train_rtx3050\weights\best.pt"
).resolve(strict=True)
print("🔒 Load model từ:", weights_path)
print("📄 File tồn tại:", os.path.exists(weights_path))

# ======================
# 🧠 Load YOLO model
# ======================
print("🧩 Đang load model trực tiếp qua torch...")
model = YOLO(str(weights_path))  # Dùng Ultralytics API chính thống
print("✅ Model YOLO load thành công!")

# ======================
# 🧠 Load CLIP model
# ======================
print("🧠 Đang tải mô hình CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP model tải thành công!")

# ======================
# 🚶 DeepSort tracker
# ======================
tracker = DeepSort(max_age=30)

# ======================
# ⚙️ Cấu hình ngưỡng và nhãn
# ======================
CONF_THRESH = 0.15
EVENT_PHONE_DISTANCE_PIX = 80
CLIP_CONF_THRESH = 0.6

CLIP_LABELS = [
    "student using phone during exam",
    "student holding phone but not using it",
    "student looking at neighbor's paper",
    "student writing on paper normally"
]

# ======================
# 🧠 Hàm tính điểm CLIP
# ======================
def clip_score(image_crop):
    inputs = clip_proc(text=CLIP_LABELS, images=image_crop, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=0).detach().cpu().numpy()
    idx = probs.argmax()
    return CLIP_LABELS[idx], float(probs[idx])

# ======================
# 🖼️ Hàm xử lý khung hình
# ======================
def process_frame(frame):
    if frame is None:
        return None

    results = model(frame, verbose=False)[0]
    print("📦 Tổng số box detect:", len(results.boxes))

    dets = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESH:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, f"id_{cls_id}")

        # ✅ In ra log để kiểm tra model có nhận đúng class không
        print(f"🎯 Phát hiện: {label} ({conf:.2f})")

        # Vẽ màu theo class
        color = (0, 0, 255) if label == "cheating" else (0, 255, 0)

        # Vẽ khung
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

# ======================
# 🌐 Giao diện Gradio
# ======================
demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(type="numpy", label="Upload Image or Frame"),  # ❌ Bỏ source="upload"
    outputs=gr.Image(type="numpy", label="Processed Frame"),
    title="📷 Exam Monitoring with YOLO + CLIP + DeepSort",
    description="Phát hiện gian lận thi cử bằng camera real-time (YOLOv8 + DeepSort + CLIP)"
)

# ======================
# 🚀 Chạy ứng dụng
# ======================
if __name__ == "__main__":
    demo.launch()
