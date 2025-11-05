import os
import cv2
import torch
torch.backends.cudnn.benchmark = True
import gradio as gr
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

# ========================
# ⚙️ CẤU HÌNH
# ========================
PROJECT_ROOT = r"D:\DaihocDaiNam\Nam4\Ki1\Chuyen_doi_so\Baitaplon\exam_monitoring_vlm"
os.chdir(PROJECT_ROOT)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 Loading YOLO...")
yolo = YOLO(r"runs\detect\train_rtx3050\weights\best.pt")
print("✅ YOLO ready.")

print("🔹 Loading BLIP VQA...")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
print(f"✅ BLIP ready on {device}")

# ========================
# 🧠 HÀM PHÂN TÍCH ẢNH
# ========================
def analyze_image(image, question):
    if image is None:
        return None, "⚠️ Hãy tải lên ảnh trước."

    # Chuyển ảnh sang OpenCV
    if isinstance(image, str):
        img_cv = cv2.imread(image)
    else:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    h, w, _ = img_cv.shape
   # YOLO detect mạnh hơn
    results = yolo.predict(
    source=img_cv,
    conf=0.6,
    iou=0.5,
    imgsz=640,
    device=device,
    verbose=False
    )[0]


    cheat_boxes = []
    cheat_index = 1

    # Vẽ box & xác định vị trí
    for box in results.boxes:
        cls = int(box.cls[0])
        label = yolo.names.get(cls, str(cls))
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Vị trí tương đối trong ảnh
        pos_x = "bên trái" if cx < w/3 else "giữa" if cx < 2*w/3 else "bên phải"
        pos_y = "hàng đầu" if cy < h/3 else "hàng giữa" if cy < 2*h/3 else "hàng sau"

        # Vẽ màu khác nhau cho từng nhãn
        if label == "cheating":
            color = (0, 0, 255)
            tag = f"cheating-{cheat_index}"
            cheat_boxes.append((pos_x, pos_y, tag))
            cheat_index += 1
        else:
            color = (0, 255, 0)
            tag = "non-cheating"

        # Vẽ khung & nhãn
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_cv, tag, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------
    # 1️⃣ Trả lời bằng YOLO nếu hỏi gian lận
    # -------------------
    lower_q = question.lower()
    if any(k in lower_q for k in ["gian lận", "cheat", "điện thoại", "phone"]):
        if not cheat_boxes:
            answer = "Không có ai gian lận hoặc dùng điện thoại."
        else:
            descs = [
                f"{i+1}. {tag} ở {y} {x}."
                for i, (x, y, tag) in enumerate(cheat_boxes)
            ]
            answer = f"Có {len(cheat_boxes)} người đang gian lận:\n" + "\n".join(descs)
        return img_cv, answer

    # -------------------
    # 2️⃣ BLIP trả lời câu hỏi chung
    # -------------------
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    inputs = blip_proc(img_pil, question, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    answer = blip_proc.decode(out[0], skip_special_tokens=True)
    return img_cv, answer


# ========================
# 💬 HÀM CHAT
# ========================
def chat_with_vlm(message, history, image):
    if image is None:
        return "⚠️ Hãy tải lên ảnh lớp học trước."
    annotated, answer = analyze_image(image, message)
    cv2.imwrite("output_temp.jpg", annotated)
    return f"🧠 **Answer:** {answer}"

# ========================
# 🌐 GIAO DIỆN
# ========================
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🧠 Chat với AI giám sát thi cử — YOLOv8 + BLIP-VQA (v3)")
    gr.Markdown("Tải ảnh phòng thi → hỏi: **Ai đang gian lận?**, **Người gian lận ở đâu?**, hoặc **Who is using the phone?**")

    with gr.Row():
        img_input = gr.Image(
        label="📷 Upload Exam Image",
        type="pil",
        image_mode="RGB",
        streaming=False,
        height=None,
        width=None,
        )

        chatbot = gr.ChatInterface(
            fn=lambda message, history, image: chat_with_vlm(message, history, image),
            additional_inputs=[img_input],
            textbox=gr.Textbox(placeholder="Nhập câu hỏi...", scale=4),
        )

if __name__ == "__main__":
    demo.launch(share=False)
