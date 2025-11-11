import os, time, gc, threading
from datetime import datetime
import cv2, torch, numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import gradio as gr

# ====== Âm thanh cảnh báo (Windows) ======
try:
    import winsound
    def beep(): winsound.Beep(2000, 700)
except Exception:
    def beep(): pass

# ========================
# ⚙️ Cấu hình
# ========================
CAM_URL = "http://192.168.1.13:4747/video"
MODEL_PERSON = "yolov8n.pt"
MODEL_CHEAT = r"runs/detect/train_rtx3050/weights/best.pt"
CONF_PERSON, CONF_CHEAT = 0.35, 0.15
EVIDENCE_DIR = "logs/evidence"
LOG_FILE = "logs/evidence_log.txt"
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

DEVICE_GPU = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 GPU: {DEVICE_GPU}")

ANALYZE_EVERY = 3
BLIP_AUTO_COOLDOWN = 10.0
BLIP_SEMAPHORE = threading.Semaphore(1)

# ========================
# 🔹 Tải mô hình
# ========================
print("🔹 Tải YOLO-person (GPU FP16)…")
yolo_person = YOLO(MODEL_PERSON).to(DEVICE_GPU)

print("🔹 Tải YOLO-cheat (CPU)…")
yolo_cheat = YOLO(MODEL_CHEAT).to("cpu")

print("🔹 Tải BLIP-2 Flan-T5-XL (CPU)…")
blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float32
).to("cpu").eval()
print("✅ Mô hình sẵn sàng.\n")

# ========================
# 🧩 Bộ phân loại text từ caption
# ========================

CHEAT_KEYWORDS = [
    "cheat", "cheating", "copy", "copying", "using phone",
    "phone", "mobile", "texting", "screen", "looking at phone",
    "using device", "talking", "communicating", "chat"
]

def predict_cheating_from_caption(caption: str) -> bool:
    text = caption.lower().strip()
    # Loại bỏ dấu câu
    for punc in [".", ",", ";", "!", "?"]:
        text = text.replace(punc, "")
    # Kiểm tra từ khóa
    return any(kw in text for kw in CHEAT_KEYWORDS)


# ========================
# 💾 Lưu bằng chứng
# ========================
def save_evidence(frame, caption=None):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(EVIDENCE_DIR, f"cheating_{ts}.jpg")
    cv2.imwrite(path, frame)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {caption or 'cheating detected'} | IMG: {path}\n")
    print(f"💾 Đã lưu bằng chứng: {path}")

# ========================
# 🔤 BLIP-2 xử lý
# ========================
def run_blip_sync(pil_img, question):
    with BLIP_SEMAPHORE:
        inputs = blip_proc(images=pil_img, text=question, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            out = blip_model.generate(**inputs, max_new_tokens=32)
        ans = blip_proc.tokenizer.decode(out[0], skip_special_tokens=True)
        del inputs, out
        gc.collect()
        return ans.strip()

def run_blip_async(pil_img, frame_bgr):
    """Chạy BLIP-2 song song, lưu bằng chứng + hiển thị text lên video"""
    def worker():
        try:
            txt = run_blip_sync(pil_img, "Describe what the student is doing.")
            if predict_cheating_from_caption(txt):
                verdict = f"🚨 Cheating confirmed: {txt}"
                color = (0, 0, 255)
                beep()
                save_evidence(frame_bgr, verdict)
            else:
                verdict = f"✅ Safe behaviour: {txt}"
                color = (0, 255, 0)

            # Ghi lên frame trước khi lưu
            cv2.putText(frame_bgr, verdict[:80], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = os.path.join(EVIDENCE_DIR, f"blip_{ts}.jpg")
            cv2.imwrite(out_path, frame_bgr)
            print(f"🧠 BLIP-2: {verdict}")
        except Exception as e:
            print("⚠️ BLIP-2 lỗi:", e)
    threading.Thread(target=worker, daemon=True).start()

# ========================
# 🎥 Phân tích video
# ========================
last_auto_blip_ts = 0.0
def analyze_frame(frame_bgr):
    global last_auto_blip_ts
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    res_person = yolo_person.predict(rgb, imgsz=640, conf=CONF_PERSON,
                                     device=DEVICE_GPU, half=True, verbose=False)[0]
    if res_person is None or res_person.boxes is None or len(res_person.boxes) == 0:
        return frame_bgr, False

    cheating = False
    for box in res_person.boxes:
        cls = int(box.cls[0])
        if yolo_person.names[cls] != "person":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = rgb[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        res_c = yolo_cheat.predict(roi, imgsz=320, conf=CONF_CHEAT, device="cpu", verbose=False)[0]
        tag, color = "non-cheating", (0, 255, 0)

        if res_c and res_c.boxes is not None and len(res_c.boxes) > 0:
            cls2 = int(res_c.boxes[0].cls[0])
            if yolo_cheat.names[cls2].lower() == "cheating":
                tag, color = "cheating", (0, 0, 255)
                cheating = True
                now = time.time()
                if now - last_auto_blip_ts >= BLIP_AUTO_COOLDOWN:
                    pil = Image.fromarray(roi)
                    run_blip_async(pil, frame_bgr.copy())
                    last_auto_blip_ts = now

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_bgr, tag, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame_bgr, cheating

# ========================
# 📡 Camera loop
# ========================
current_frame = None
def camera_loop():
    global current_frame
    cap = cv2.VideoCapture(CAM_URL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if not cap.isOpened():
        print("⚠️ Không thể mở camera.")
        return
    print("📷 Camera OK!")

    frame_idx, analyzed_frame, cheat_flag = 0, None, False
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        current_frame = frame.copy()
        frame_idx += 1
        if frame_idx % ANALYZE_EVERY == 0:
            analyzed_frame, cheat_flag = analyze_frame(frame.copy())

        display = analyzed_frame if analyzed_frame is not None else frame
        status = "CHEATING" if cheat_flag else "SAFE"
        color = (0, 0, 255) if cheat_flag else (0, 255, 0)
        cv2.putText(display, f"STATUS: {status}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.imshow("📡 Exam Monitor Hybrid (YOLO + BLIP-2)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ========================
# 💬 Chat BLIP-2
# ========================
def chat_with_vlm(message, history):
    global current_frame
    if current_frame is None:
        return {"role": "assistant", "content": "⚠️ Chưa có khung hình camera."}

    # Chuyển frame sang PIL để BLIP-2 xử lý
    rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    ans = run_blip_sync(pil, message)
    verdict_is_cheat = predict_cheating_from_caption(ans)

    # Ghi chú đánh giá
    verdict_text = "🚨 Cheating" if verdict_is_cheat else "✅ Safe"
    color = (0, 0, 255) if verdict_is_cheat else (0, 255, 0)

    # Lưu ảnh bằng chứng (và ghi log)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"chat_{'cheat' if verdict_is_cheat else 'safe'}_{ts}.jpg"
    out_path = os.path.join(EVIDENCE_DIR, filename)

    frame_copy = current_frame.copy()
    cv2.putText(frame_copy, f"{verdict_text}: {ans[:80]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(out_path, frame_copy)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {verdict_text}: {ans} | IMG: {out_path}\n")

    print(f"💾 Lưu bằng chứng chat: {out_path}")
    return {"role": "assistant", "content": f"🧠 {ans}\n\n**Đánh giá:** {verdict_text}"}
def chat_with_vlm(message, history):
    global current_frame
    if current_frame is None:
        return {"role": "assistant", "content": "⚠️ Chưa có khung hình camera."}

    # Chuyển frame sang PIL để BLIP-2 xử lý
    rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    ans = run_blip_sync(pil, message)
    verdict_is_cheat = predict_cheating_from_caption(ans)

    # Ghi chú đánh giá
    verdict_text = "🚨 Cheating" if verdict_is_cheat else "✅ Safe"
    color = (0, 0, 255) if verdict_is_cheat else (0, 255, 0)

    # Lưu ảnh bằng chứng (và ghi log)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"chat_{'cheat' if verdict_is_cheat else 'safe'}_{ts}.jpg"
    out_path = os.path.join(EVIDENCE_DIR, filename)

    frame_copy = current_frame.copy()
    cv2.putText(frame_copy, f"{verdict_text}: {ans[:80]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(out_path, frame_copy)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {verdict_text}: {ans} | IMG: {out_path}\n")

    print(f"💾 Lưu bằng chứng chat: {out_path}")
    return {"role": "assistant", "content": f"🧠 {ans}\n\n**Đánh giá:** {verdict_text}"}

# ========================
# 🚀 Tạo giao diện Gradio
# ========================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🤖 Exam Monitoring — YOLO + BLIP-2 Hybrid (640p)")
    gr.ChatInterface(
        fn=chat_with_vlm,
        title="BLIP-2 Chat (khung hình hiện tại)",
        textbox=gr.Textbox(placeholder="Ví dụ: 'Is student cheating?'", lines=1),
        type="messages",
    )

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    demo.launch(share=False)