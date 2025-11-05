import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import gradio as gr
import threading
import time
import os

# ========================
# ⚙️ CẤU HÌNH
# ========================
CAM_URL = "http://172.16.31.141:4747/video"   # 👉 Thay IP điện thoại IP Webcam của bạn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs/evidence"
os.makedirs(LOG_DIR, exist_ok=True)

print(f"🧠 Thiết bị: {DEVICE}")

# ========================
# 🔹 TẢI MÔ HÌNH
# ========================
print("🔹 Đang tải YOLO...")
yolo = YOLO(r"runs\detect\train_rtx3050\weights\best.pt")  # Đường dẫn model YOLO của bạn
print("✅ YOLO sẵn sàng.")

print("🔹 Đang tải BLIP-2 Flan-T5-XL")
blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
print("✅ BLIP-2 Flan-T5-XL sẵn sàng.")

# ========================
# 🧠 HÀM PHÂN TÍCH ẢNH
# ========================
def analyze_frame(frame, question):
    """Phân tích frame và trả lời câu hỏi bằng YOLO + BLIP-2 Flan-T5-XL."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo.predict(source=rgb_frame, conf=0.15, verbose=False)[0]

    h, w, _ = frame.shape
    cheat_boxes = []
    cheat_index = 1

    for box in results.boxes:
        cls = int(box.cls[0])
        label = yolo.names.get(cls, str(cls))
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pos_x = "bên trái" if cx < w / 3 else "giữa" if cx < 2 * w / 3 else "bên phải"
        pos_y = "hàng đầu" if cy < h / 3 else "hàng giữa" if cy < 2 * h / 3 else "hàng sau"

        if label == "cheating":
            color = (0, 0, 255)
            tag = f"cheating-{cheat_index}"
            cheat_boxes.append((pos_x, pos_y, tag, conf))
            cheat_index += 1
        else:
            color = (0, 255, 0)
            tag = "non-cheating"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{tag} ({conf:.2f})", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------
    # 1️⃣ Nếu hỏi về gian lận
    # -------------------
    lower_q = question.lower()
    if any(k in lower_q for k in ["gian lận", "cheat", "điện thoại", "phone"]):
        if not cheat_boxes:
            return "Không có ai gian lận hoặc dùng điện thoại."
        else:
            descs = [f"{i+1}. {tag} ({conf:.2f}) ở {y} {x}."
                     for i, (x, y, tag, conf) in enumerate(cheat_boxes)]
            answer = f"🚨 Có {len(cheat_boxes)} người đang gian lận:\n" + "\n".join(descs)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(LOG_DIR, f"cheating_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"📸 Ảnh bằng chứng đã lưu: {save_path}")
            return answer

    # -------------------
    # 2️⃣ Câu hỏi khác → BLIP-2 Flan-T5-XL
    # -------------------
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = blip_proc(images=img_pil, text=question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=40)
    answer = blip_proc.tokenizer.decode(out[0], skip_special_tokens=True)

    torch.cuda.empty_cache()
    return answer


# ========================
# 💬 CHAT GRADIO
# ========================
def chat_with_vlm(message, history):
    global current_frame
    if current_frame is None:
        return {"role": "assistant", "content": "⚠️ Chưa có khung hình camera."}
    answer = analyze_frame(current_frame.copy(), message)
    return {"role": "assistant", "content": f"🧠 {answer}"}


# ========================
# 🎥 LUỒNG CAMERA OPENCV
# ========================
current_frame = None

def camera_loop():
    global current_frame
    cap = cv2.VideoCapture(CAM_URL)
    if not cap.isOpened():
        print("⚠️ Không thể kết nối tới camera. Kiểm tra lại IP Webcam.")
        return

    print("📷 Camera đã kết nối thành công!")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        current_frame = frame.copy()
        cv2.imshow("📡 Luồng camera realtime (Nhấn Q để thoát)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ========================
# 🌐 GIAO DIỆN GRADIO
# ========================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🤖 AI Giám sát thi cử (YOLOv8 + BLIP-2 Flan-T5-XL)")
    gr.Markdown("Camera realtime hiển thị qua OpenCV — đặt câu hỏi cho AI tại đây 👇")

    chatbot = gr.ChatInterface(
        fn=chat_with_vlm,
        title="AI Giám sát thi cử",
        textbox=gr.Textbox(placeholder="Nhập câu hỏi: Ai đang gian lận?..."),
        type="messages",
    )

# ========================
# 🚀 CHẠY SONG SONG
# ========================
if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    demo.launch(share=False)
