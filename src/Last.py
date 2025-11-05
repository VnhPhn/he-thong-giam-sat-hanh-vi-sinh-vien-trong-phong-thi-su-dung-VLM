# exam_guard.py
import os
import time
import gc
import math
import threading
from datetime import datetime

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import gradio as gr

# ====== (Windows) Cảnh báo âm thanh
try:
    import winsound
    def beep(): winsound.Beep(2000, 700)
except Exception:
    def beep(): pass  # no-op trên non-Windows

# ========================
# ⚙️ CẤU HÌNH
# ========================
CAM_URL = "http://172.16.15.0:4747/video"     # 👉 IP Webcam của bạn (DroidCam/IP Webcam)
YOLO_WEIGHTS = r"runs/detect/train_rtx3050/weights/best.pt"  # hoặc yolov8s.pt để test
CONF_DET = 0.01          # Ngưỡng YOLO
BLIP_MAX_NEW_TOKENS = 32 # Độ dài câu trả lời
EVIDENCE_DIR = "logs/evidence"
LOG_FILE = "logs/evidence_log.txt"
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"🧠 Thiết bị: {DEVICE}, dtype: {DTYPE}")

# ========================
# 🔹 TẢI MÔ HÌNH
# ========================
print("🔹 Đang tải YOLO...")
yolo = YOLO(YOLO_WEIGHTS)
yolo.to(DEVICE)
print("✅ YOLO sẵn sàng.")

print("🔹 Đang tải BLIP-2 Flan-T5-XL...")
blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=DTYPE
).to(DEVICE)
blip_model.eval()
print("✅ BLIP-2 Flan-T5-XL sẵn sàng.")

# ========================
# 🔤 TIỆN ÍCH & NGƯỠNG
# ========================
SUS_KEYWORDS = [
    "cheating", "copying", "looking at another paper",
    "using a phone", "phone", "mobile", "device", "texting",
    "peeking", "whispering", "passing paper"
]

# Các tên lớp khả dụng tùy bộ dữ liệu của bạn
# Bạn có thể sửa lại cho trùng khớp dataset
CHEAT_LABELS = {"cheating"}         # Nếu mô hình có lớp 'cheating'
PERSON_LABELS = {"person", "student", "pupil"}
PHONE_LABELS = {"phone", "cell phone", "mobile", "smartphone"}

def iou(a, b):
    # a, b: [x1, y1, x2, y2]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def expand_box(x1, y1, x2, y2, scale, w, h):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1 = int(max(0, cx - bw/2)), int(max(0, cy - bh/2))
    nx2, ny2 = int(min(w-1, cx + bw/2)), int(min(h-1, cy + bh/2))
    return nx1, ny1, nx2, ny2

def run_blip_question(pil_img: Image.Image, question: str) -> str:
    inputs = blip_proc(images=pil_img, text=question, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            out = blip_model.generate(**inputs, max_new_tokens=BLIP_MAX_NEW_TOKENS)
    ans = blip_proc.tokenizer.decode(out[0], skip_special_tokens=True)
    # Thu dọn VRAM
    del inputs, out
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return ans.strip()

def looks_suspicious(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in SUS_KEYWORDS)

# ========================
# 🧠 PHÂN TÍCH FRAME: YOLO + BLIP trên ROI nghi ngờ
# ========================
def analyze_frame_fused(frame_bgr):
    """
    - YOLO phát hiện person/phone/cheating
    - Nếu có 'cheating' → xác nhận ngay
    - Nếu có person & phone gần nhau → BLIP hỏi xác minh trên ROI ghép
    """
    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # YOLO predict
    results = yolo.predict(source=rgb, conf=CONF_DET, verbose=False)[0]
    if results is None or results.boxes is None or len(results.boxes) == 0:
        return frame_bgr, False, None

    boxes = []
    persons = []
    phones = []
    cheating_boxes = []

    # Thu thập bbox theo nhãn
    for b in results.boxes:
        cls_id = int(b.cls[0])
        label = yolo.names[cls_id] if hasattr(yolo, "names") else str(cls_id)
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        boxes.append((label, conf, (x1, y1, x2, y2)))

        if label.lower() in PERSON_LABELS:
            persons.append((conf, (x1, y1, x2, y2)))
        if label.lower() in PHONE_LABELS:
            phones.append((conf, (x1, y1, x2, y2)))
        if label.lower() in CHEAT_LABELS:
            cheating_boxes.append((conf, (x1, y1, x2, y2)))

    # Vẽ tất cả bbox
    for label, conf, (x1, y1, x2, y2) in boxes:
        color = (0, 255, 0)
        if label.lower() in PHONE_LABELS: color = (255, 255, 0)
        if label.lower() in CHEAT_LABELS: color = (0, 0, 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_bgr, f"{label} {conf:.2f}", (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Nếu YOLO đã có 'cheating' → xác minh nhẹ bằng BLIP (toàn ROI) để lấy mô tả
    if cheating_boxes:
        # Lấy box có conf cao nhất
        cheating_boxes.sort(key=lambda x: x[0], reverse=True)
        _, (x1, y1, x2, y2) = cheating_boxes[0]
        rx1, ry1, rx2, ry2 = expand_box(x1, y1, x2, y2, 1.1, w, h)
        roi = rgb[ry1:ry2, rx1:rx2]
        pil = Image.fromarray(roi)
        caption = run_blip_question(pil, "Describe the suspicious behavior in this image in one sentence.")
        return frame_bgr, True, caption

    # Nếu không có 'cheating', nhưng có person + phone → kiểm tra gần nhau rồi BLIP hỏi
    suspicious_caption = None
    S_MAX_DIST_PIX = max(40, int(0.08 * max(w, h)))  # ngưỡng gần nhau (tương đối theo kích thước ảnh)
    for p_conf, (px1, py1, px2, py2) in persons:
        pcx, pcy = (px1+px2)//2, (py1+py2)//2
        for ph_conf, (hx1, hy1, hx2, hy2) in phones:
            hcx, hcy = (hx1+hx2)//2, (hy1+hy2)//2
            dist = math.hypot(pcx - hcx, pcy - hcy)
            if dist <= S_MAX_DIST_PIX:
                # Gộp ROI person + phone
                gx1, gy1 = min(px1, hx1), min(py1, hy1)
                gx2, gy2 = max(px2, hx2), max(py2, hy2)
                gx1, gy1, gx2, gy2 = expand_box(gx1, gy1, gx2, gy2, 1.1, w, h)
                roi = rgb[gy1:gy2, gx1:gx2]
                pil = Image.fromarray(roi)
                ans = run_blip_question(
                    pil,
                    "Is the student cheating in the exam? Answer briefly (e.g., 'using a phone', 'copying', or 'no')."
                )
                if looks_suspicious(ans):
                    suspicious_caption = ans
                    # Viền đỏ vùng nghi ngờ
                    cv2.rectangle(frame_bgr, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                    cv2.putText(frame_bgr, "suspicious", (gx1, max(0, gy1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    return frame_bgr, True, suspicious_caption

    return frame_bgr, False, None

# ========================
# 🎥 CAMERA LOOP + CẢNH BÁO
# ========================
current_frame_bgr = None
last_alert_ts = 0.0
ALERT_COOLDOWN = 2.0  # giây, tránh kêu chuông liên tục

def save_evidence(frame_bgr, caption: str | None):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    img_path = os.path.join(EVIDENCE_DIR, f"cheating_{ts}.jpg")
    cv2.imwrite(img_path, frame_bgr)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERT -> {caption or 'cheating detected'} | IMG: {img_path}\n")
    print(f"💾 Lưu bằng chứng: {img_path} | Mô tả: {caption}")

def camera_loop():
    global current_frame_bgr, last_alert_ts
    cap = cv2.VideoCapture(CAM_URL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("⚠️ Không thể kết nối tới camera. Kiểm tra lại IP Webcam.")
        return
    print("📷 Camera đã kết nối thành công!")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (640, 480))
        current_frame_bgr = frame.copy()

        # Phân tích frame (YOLO + BLIP ROI)
        analyzed, is_cheat, caption = analyze_frame_fused(frame)

        # Cảnh báo
        now = time.time()
        if is_cheat and (now - last_alert_ts >= ALERT_COOLDOWN):
            beep()
            save_evidence(analyzed, caption)
            last_alert_ts = now

        cv2.imshow("📡 Giám sát thi cử (Realtime) — nhấn 'q' để thoát", analyzed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================
# 💬 BLIP-2 Q&A TRÊN KHUNG HÌNH HIỆN TẠI
# ========================
def chat_with_vlm(message, history):
    global current_frame_bgr
    if current_frame_bgr is None:
        return {"role": "assistant", "content": "⚠️ Chưa có khung hình camera."}

    # Hỏi trên toàn khung hình hiện tại (để người dùng tự do Q&A)
    rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    ans = run_blip_question(pil, message)
    return {"role": "assistant", "content": f"🧠 {ans}"}

# ========================
# 🌐 GIAO DIỆN GRADIO
# ========================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🤖 AI Giám sát thi cử — YOLOv8 + BLIP-2 Flan-T5-XL")
    gr.Markdown(
        "• Cửa sổ OpenCV hiển thị camera realtime\n"
        "• Tự động cảnh báo + lưu bằng chứng khi xác nhận gian lận (YOLO + BLIP-2)\n"
        "• Hỏi-đáp về khung hình hiện tại ở khung chat bên dưới"
    )
    gr.ChatInterface(
        fn=chat_with_vlm,
        title="BLIP-2 Q&A (khung hình hiện tại)",
        textbox=gr.Textbox(placeholder="Ví dụ: 'Is anyone using a phone?'", lines=1),
        type="messages",
    )

# ========================
# 🚀 CHẠY SONG SONG
# ========================
if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    demo.launch(share=False)
