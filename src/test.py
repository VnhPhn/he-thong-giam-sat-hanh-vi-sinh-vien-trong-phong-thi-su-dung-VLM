import os
import time
import gc
import threading
from datetime import datetime

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import gradio as gr

# ====== Âm thanh cảnh báo (Windows) ======
try:
    import winsound
    def beep():
        winsound.Beep(2000, 700)
except Exception:
    def beep():
        pass

# ========================
# ⚙️ CẤU HÌNH
# ========================
CAM_URL = "http://192.168.1.13:4747/video"

MODEL_PERSON = "yolov8n.pt"  # YOLO COCO: có class "person"
MODEL_CHEAT = r"runs/detect/train_rtx3050/weights/best.pt"  # mô hình 2 lớp cheating / non-cheating của bạn

CONF_PERSON = 0.35   # ngưỡng phát hiện người
CONF_CHEAT = 0.15    # ngưỡng cheat / non-cheat

EVIDENCE_DIR = "logs/evidence"
LOG_FILE = "logs/evidence_log.txt"
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

DEVICE_GPU = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Thiết bị GPU chính: {DEVICE_GPU}")

ANALYZE_EVERY = 2        # chỉ chạy YOLO mỗi 2 frame (giữ FPS)
BLIP_AUTO_COOLDOWN = 6.0 # giây, auto mô tả sau mỗi lần cheat

# ========================
# 🔹 TẢI MÔ HÌNH
# ========================
print("🔹 Tải YOLO-person (COCO, GPU)…")
yolo_person = YOLO(MODEL_PERSON)
yolo_person.to(DEVICE_GPU)

print("🔹 Tải YOLO-cheat (2 lớp, CPU)…")
yolo_cheat = YOLO(MODEL_CHEAT)
yolo_cheat.to("cpu")

print("🔹 Tải BLIP-2 Flan-T5-XL (CPU)…")
blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float32
).to("cpu").eval()
print("✅ Models ready.")

# ========================
# 🔤 BLIP-2 util
# ========================
BLIP_MAX_NEW_TOKENS = 32

def run_blip_sync(pil_img: Image.Image, question: str) -> str:
    """Gọi BLIP-2 đồng bộ (dùng cho chat)."""
    inputs = blip_proc(images=pil_img, text=question, return_tensors="pt").to("cpu")
    with torch.inference_mode():
        out = blip_model.generate(**inputs, max_new_tokens=BLIP_MAX_NEW_TOKENS)
    ans = blip_proc.tokenizer.decode(out[0], skip_special_tokens=True)
    del inputs, out
    gc.collect()
    return ans.strip()

def run_blip_async(pil_img: Image.Image, question: str):
    """Gọi BLIP-2 nền khi auto mô tả, không block FPS."""
    def worker():
        try:
            txt = run_blip_sync(pil_img, question)
            # Lưu log mô tả cho vui, nếu muốn có thể lưu ra file / DB ở đây
            print(f"🧠 [BLIP-2 auto] {txt}")
        except Exception as e:
            print("⚠️ Lỗi BLIP-2 auto:", e)
    threading.Thread(target=worker, daemon=True).start()

# ========================
# 💾 Lưu bằng chứng
# ========================
def save_evidence(frame_bgr, caption: str | None):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(EVIDENCE_DIR, f"cheating_{ts}.jpg")
    cv2.imwrite(path, frame_bgr)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ALERT -> "
            f"{caption or 'cheating detected'} | IMG: {path}\n"
        )
    print(f"💾 Lưu bằng chứng: {path}")

# ========================
# 🧠 BIẾN TOÀN CỤC
# ========================
current_frame_bgr = None         # dùng cho chat BLIP-2
last_fps = 0.0
last_auto_blip_ts = 0.0

# ========================
# 🔍 PHÂN TÍCH MỘT FRAME
# ========================
def analyze_frame(frame_bgr):
    """Phát hiện person (GPU), cheat (CPU), vẽ khung + trả bool gian lận."""
    global last_auto_blip_ts

    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # B1. YOLO-person trên GPU
    with torch.inference_mode():
        res_person = yolo_person.predict(
            source=rgb,
            imgsz=640,
            conf=CONF_PERSON,
            device=DEVICE_GPU,
            half=(DEVICE_GPU == "cuda"),
            verbose=False
        )[0]

    cheating_detected = False

    if res_person is None or res_person.boxes is None or len(res_person.boxes) == 0:
        return frame_bgr, cheating_detected

    for box in res_person.boxes:
        cls_id = int(box.cls[0])
        label = yolo_person.names[cls_id]
        if label != "person":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = rgb[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # B2. YOLO-cheat trên CPU (2 lớp)
        with torch.inference_mode():
            res_cheat = yolo_cheat.predict(
                source=roi,
                imgsz=320,
                conf=CONF_CHEAT,
                device="cpu",
                verbose=False
            )[0]

        tag = "non-cheating"
        color = (0, 255, 0)

        if res_cheat is not None and res_cheat.boxes is not None and len(res_cheat.boxes) > 0:
            # Lấy box có conf cao nhất
            res_cheat.boxes = res_cheat.boxes[res_cheat.boxes.conf.argsort(descending=True)]
            cls2 = int(res_cheat.boxes[0].cls[0])
            label2 = yolo_cheat.names[cls2].lower()

            if label2 == "cheating":
                tag = "cheating"
                color = (0, 0, 255)
                cheating_detected = True

                now = time.time()
                # Auto BLIP + lưu bằng chứng theo cooldown
                if now - last_auto_blip_ts >= BLIP_AUTO_COOLDOWN:
                    pil = Image.fromarray(roi)
                    run_blip_async(pil, "Describe what the student is doing in this image.")
                    save_evidence(frame_bgr, "cheating by YOLO + BLIP-2 auto")
                    last_auto_blip_ts = now

        # Vẽ khung quanh person
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_bgr,
            tag,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return frame_bgr, cheating_detected

# ========================
# 🎥 CAMERA LOOP
# ========================
def camera_loop():
    global current_frame_bgr, last_fps

    cap = cv2.VideoCapture(CAM_URL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # giới hạn FPS input để GPU đỡ căng

    if not cap.isOpened():
        print("⚠️ Không thể kết nối camera.")
        return

    print("📷 Camera đã kết nối!")

    frame_idx = 0
    analyzed_frame = None
    last_status_cheat = False

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        current_frame_bgr = frame.copy()
        frame_idx += 1

        # Chỉ phân tích mỗi N frame để đỡ tốn tài nguyên
        if frame_idx % ANALYZE_EVERY == 0:
            t0 = time.time()
            analyzed, is_cheat = analyze_frame(frame.copy())
            dt = time.time() - t0
            last_fps = 1.0 / dt if dt > 0 else 0.0
            analyzed_frame = analyzed
            last_status_cheat = is_cheat
        else:
            # Dùng lại frame đã phân tích gần nhất
            if analyzed_frame is None:
                analyzed_frame = frame.copy()
                last_status_cheat = False

        # Overlay trạng thái & FPS
        overlay = analyzed_frame.copy()
        status_text = "DETECTED: CHEATING" if last_status_cheat else "STATUS: SAFE"
        color = (0, 0, 255) if last_status_cheat else (0, 255, 0)
        cv2.putText(
            overlay,
            status_text,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3
        )
        cv2.putText(
            overlay,
            f"FPS: {last_fps:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("📡 Exam Monitor (Dual YOLO + BLIP-2 Chat)", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================
# 💬 GRADIO CHAT VỚI BLIP-2
# ========================
def chat_with_vlm(message, history):
    global current_frame_bgr
    if current_frame_bgr is None:
        return {"role": "assistant", "content": "⚠️ Chưa có khung hình camera để phân tích."}

    rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # Chat này cho BLIP chạy đồng bộ (trên CPU), nhưng không ảnh hưởng GPU
    try:
        ans = run_blip_sync(pil, message)
        return {"role": "assistant", "content": f"🧠 {ans}"}
    except Exception as e:
        return {"role": "assistant", "content": f"⚠️ BLIP-2 lỗi: {e}"}

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🤖 Exam Monitoring — Dual YOLO + BLIP-2 Chat (640p)")
    gr.Markdown(
        "- Cửa sổ OpenCV hiển thị camera realtime\n"
        "- YOLOv8n (GPU) phát hiện người, YOLO custom (CPU) đánh giá cheating / non-cheating\n"
        "- BLIP-2 tự mô tả khi phát hiện cheating (chạy nền, không tụt FPS)\n"
        "- Bạn có thể hỏi thêm về khung hình hiện tại ở phần chat bên dưới"
    )
    gr.ChatInterface(
        fn=chat_with_vlm,
        title="BLIP-2 Q&A (khung hình hiện tại)",
        textbox=gr.Textbox(placeholder="Ví dụ: 'Is anyone cheating?'", lines=1),
        type="messages",
    )

# ========================
# 🚀 MAIN
# ========================
if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    demo.launch(share=False)
