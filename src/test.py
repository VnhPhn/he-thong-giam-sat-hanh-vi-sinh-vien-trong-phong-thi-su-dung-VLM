# ========================================
# 📘 predict_image_cheat_only.py — Chỉ đóng khung người gian lận + xác suất %
# ========================================

import os, gc, torch, cv2, math
import numpy as np
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ========================
# ⚙️ CẤU HÌNH
# ========================
IMG_PATH = "data/test/hq720.jpg"   # 👉 Ảnh cần dự đoán
YOLO_WEIGHTS = r"runs/detect/train_rtx3050/weights/best.pt"
CONF_DET = 0.2

EVIDENCE_DIR = "logs/evidence"
LOG_FILE = os.path.join("logs", "evidence_log.txt")
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"🧠 Thiết bị: {DEVICE}, dtype: {DTYPE}")

# ========================
# 🔹 TẢI MÔ HÌNH
# ========================
print("🔹 Đang tải YOLO...")
yolo = YOLO(YOLO_WEIGHTS).to(DEVICE)
print("✅ YOLO sẵn sàng.")

print("🔹 Đang tải BLIP-2 Flan-T5-XL...")
blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=DTYPE
).to(DEVICE)
blip_model.eval()
print("✅ BLIP-2 sẵn sàng.\n")

# ========================
# 🔤 TỪ KHÓA NGHI NGỜ
# ========================
SUS_KEYWORDS = [
    "cheating", "copying", "using a phone", "phone",
    "mobile", "device", "texting", "peeking", "passing paper"
]

# ========================
# 🧩 HÀM TIỆN ÍCH
# ========================
def run_blip(img_pil, question):
    """Sinh câu trả lời từ BLIP-2"""
    inputs = blip_proc(images=img_pil, text=question, return_tensors="pt").to(DEVICE)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
        out = blip_model.generate(**inputs, max_new_tokens=32)
    ans = blip_proc.tokenizer.decode(out[0], skip_special_tokens=True)
    del inputs, out
    if DEVICE == "cuda": torch.cuda.empty_cache()
    gc.collect()
    return ans.strip()

def looks_suspicious(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in SUS_KEYWORDS)

def expand_box(x1, y1, x2, y2, scale, w, h):
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1)*scale, (y2-y1)*scale
    nx1, ny1 = int(max(0, cx-bw/2)), int(max(0, cy-bh/2))
    nx2, ny2 = int(min(w-1, cx+bw/2)), int(min(h-1, cy+bh/2))
    return nx1, ny1, nx2, ny2

# ========================
# 🔍 DỰ ĐOÁN ẢNH
# ========================
print(f"📸 Phân tích ảnh: {IMG_PATH}")
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise SystemExit(f"❌ Không tìm thấy ảnh: {IMG_PATH}")
h, w, _ = img_bgr.shape

results = yolo.predict(source=img_bgr, conf=CONF_DET, verbose=False)[0]
persons, phones, cheating_boxes = [], [], []

for b in results.boxes:
    label = yolo.names[int(b.cls[0])].lower()
    conf = float(b.conf[0])
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    if label == "person": persons.append((conf, (x1, y1, x2, y2)))
    if label in ["phone", "mobile", "smartphone"]: phones.append((conf, (x1, y1, x2, y2)))
    if label == "cheating": cheating_boxes.append((conf, (x1, y1, x2, y2)))

# ========================
# 🧠 PHÂN TÍCH GIAN LẬN
# ========================
suspected_region = None
prob_percent = 0
caption = ""

# TH1: YOLO đã có nhãn 'cheating'
if cheating_boxes:
    cheating_boxes.sort(key=lambda x: x[0], reverse=True)
    conf, (x1, y1, x2, y2) = cheating_boxes[0]
    suspected_region = (x1, y1, x2, y2)
    prob_percent = int(conf * 100)
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size > 0:
        img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        caption = run_blip(img_pil, "Describe the suspicious behavior in this image.")
    else:
        caption = "Suspicious activity detected."

# TH2: Không có nhãn cheating → Kiểm tra person + phone gần nhau
elif persons and phones:
    persons.sort(key=lambda x: x[0], reverse=True)
    phones.sort(key=lambda x: x[0], reverse=True)
    p_conf, (px1, py1, px2, py2) = persons[0]
    ph_conf, (hx1, hy1, hx2, hy2) = phones[0]
    pcx, pcy = (px1+px2)//2, (py1+py2)//2
    hcx, hcy = (hx1+hx2)//2, (hy1+hy2)//2
    dist = math.hypot(pcx - hcx, pcy - hcy)

    # Nếu khoảng cách nhỏ → nghi ngờ
    if dist < max(w, h) * 0.2:
        gx1, gy1 = min(px1, hx1), min(py1, hy1)
        gx2, gy2 = max(px2, hx2), max(py2, hy2)
        gx1, gy1, gx2, gy2 = expand_box(gx1, gy1, gx2, gy2, 1.1, w, h)
        roi = img_bgr[gy1:gy2, gx1:gx2]
        if roi.size > 0:
            img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            blip_ans = run_blip(img_pil, "Is the student cheating? Answer briefly.")
            caption = blip_ans
            if looks_suspicious(blip_ans):
                suspected_region = (gx1, gy1, gx2, gy2)
                prob_percent = int(((p_conf + ph_conf) / 2) * 100)
            else:
                prob_percent = 0

# ========================
# 🎯 VẼ KHUNG NGHI NGỜ
# ========================
if suspected_region and prob_percent > 0:
    x1, y1, x2, y2 = map(int, suspected_region)
    label = f"Cheating: {prob_percent}%"
    # Vẽ khung đỏ
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 4)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (0, 0, 255), -1)
    cv2.putText(img_bgr, label, (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
else:
    caption = caption or "No cheating detected."

print(f"🧠 Mô tả hành vi: {caption}")
print(f"📊 Xác suất gian lận: {prob_percent}%")

# ========================
# 💾 LƯU KẾT QUẢ
# ========================
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out_path = os.path.join(EVIDENCE_DIR, f"cheating_{timestamp}.jpg")
cv2.imwrite(out_path, img_bgr)
with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(f"[{datetime.now()}] {IMG_PATH} -> {prob_percent}% | {caption} | IMG: {out_path}\n")

print(f"💾 Đã lưu ảnh: {out_path}")
print(f"🗒️ Log ghi tại: {LOG_FILE}")

cv2.imshow("🧠 Kết quả phát hiện gian lận", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
