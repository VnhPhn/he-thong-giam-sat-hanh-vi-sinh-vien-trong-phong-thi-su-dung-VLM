<h2 align="center">
  <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
  🎓 KHOA CÔNG NGHỆ THÔNG TIN (ĐẠI HỌC ĐẠI NAM) 🎓
  </a>
</h2>

<h2 align="center">
   XÂY DỰNG HỆ THỐNG PHÂN TÍCH GIÁM SÁT HÀNH VI CỦA SINH VIÊN TRONG PHÒNG THI <br>
   DỰA TRÊN MÔ HÌNH NGÔN NGỮ THỊ GIÁC (VLM)
</h2>

<div align="center">
  <p align="center">
    <img width="200" alt="dnu_logo" src="https://github.com/user-attachments/assets/2bcb1a6c-774c-4e7d-b14d-8c53dbb4067f" />
    <img width="180" alt="fitdnu_logo" src="https://github.com/user-attachments/assets/ec4815af-e477-480b-b9fa-c490b74772b8" />
    <img width="170" alt="aiotlab_logo" src="https://github.com/user-attachments/assets/41ef702b-3d6e-4ac4-beac-d8c9a874bca9" />
  </p>

  <p align="center">
    <a href="https://dainam.edu.vn">
      <img alt="DaiNam University"
           src="https://img.shields.io/badge/DaiNam_University-ff6b35?style=flat-square">
    </a>
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
      <img alt="Faculty of IT"
           src="https://img.shields.io/badge/Faculty_of_IT-0066cc?style=flat-square">
    </a>
    <a href="https://www.facebook.com/DNUAIoTLab">
      <img alt="AIoTLab"
           src="https://img.shields.io/badge/AIoTLab-28a745?style=flat-square&logo=facebook&logoColor=white">
    </a>
  </p>

  <p align="center">
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"></a>
    <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
    <a href="https://ultralytics.com/yolov8"><img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-00FFFF?style=flat-square"></a>
    <a href="https://huggingface.co/Salesforce/blip2-flan-t5-xl"><img alt="BLIP-2" src="https://img.shields.io/badge/BLIP--2-FED141?style=flat-square&logo=huggingface&logoColor=black"></a>
    <a href="https://flask.palletsprojects.com/"><img alt="Flask" src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white"></a>
    <a href="https://gradio.app/"><img alt="Gradio" src="https://img.shields.io/badge/Gradio-FF7F50?style=flat-square"></a>
    <a href="https://opencv.org/"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-27338e?style=flat-square&logo=opencv&logoColor=white"></a>
    <img alt="DeepSort" src="https://img.shields.io/badge/DeepSort-5A5A5A?style=flat-square">
    <img alt="CUDA" src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white">
    <img alt="VLM Integration" src="https://img.shields.io/badge/VLM_Integration-4C8EDA?style=flat-square">
  </p>

  <p align="center">
    <img alt="Made with Love" src="https://img.shields.io/badge/Made_with-❤️-ff69b4?style=flat-square">
    <img alt="Status: Demo ready" src="https://img.shields.io/badge/Status-Demo_ready-00c853?style=flat-square">
    <img alt="Language: Vietnamese" src="https://img.shields.io/badge/Language-Vietnamese-22b8cf?style=flat-square">
  </p>
</div>

---

## 🧭 GIỚI THIỆU HỆ THỐNG

Đề tài tập trung vào việc **phân tích và giám sát hành vi của sinh viên trong phòng thi** bằng cách kết hợp giữa **mô hình thị giác máy tính (YOLOv8)** và **mô hình ngôn ngữ thị giác (VLM – BLIP-2)**.  

Hệ thống được xây dựng nhằm:
- 🎥 Giám sát **video thời gian thực** từ camera IP hoặc webcam.  
- 🔍 Phát hiện **hành vi khả nghi** như: sử dụng điện thoại, nhìn bài người khác, gian lận.  
- 🧠 Phân tích ngữ nghĩa hình ảnh bằng **BLIP-2 Flan-T5-XL** để xác nhận hành vi.  
- 🚨 Phát cảnh báo âm thanh khi phát hiện gian lận.  
- 💾 Lưu lại **ảnh bằng chứng** cùng thời gian và xác suất gian lận.  

---

## 🎯 MỤC TIÊU
- Phát hiện **tự động** hành vi gian lận trong phòng thi.  
- Kết hợp giữa **YOLOv8 (thị giác)** và **BLIP-2 (ngữ nghĩa)**.  
- Sinh báo cáo gồm: ảnh, mô tả hành vi, xác suất, thời gian.  
- Giao diện hiển thị trực quan (Gradio / Flask).  
- Ứng dụng trong các **đồ án chuyển đổi số, nghiên cứu AI giám sát thông minh**.

---

## ⚙️ CÔNG NGHỆ SỬ DỤNG

| Thành phần | Công nghệ | Vai trò |
|-------------|------------|----------|
| Phát hiện đối tượng | **YOLOv8 (Ultralytics)** | Nhận diện người, điện thoại, hành vi bất thường |
| Phân tích ngữ nghĩa ảnh | **BLIP-2 Flan-T5-XL (HuggingFace)** | Mô tả và hiểu ngữ cảnh hành vi |
| Xử lý video | **OpenCV + NumPy** | Đọc luồng, trích khung hình, vẽ bounding box |
| Theo dõi đối tượng | **DeepSort** | Gán ID và theo dõi sinh viên trong khung hình |
| Giao diện demo | **Gradio / Flask** | Hiển thị, tương tác và chạy thử hệ thống |
| Cảnh báo | **winsound (Windows)** | Phát âm báo khi phát hiện gian lận |
| Lưu bằng chứng | **datetime + os** | Lưu log + ảnh vào thư mục `logs/evidence/` |

---

## 🧩 KIẾN TRÚC HỆ THỐNG

```text
![Sơ đồ kiến trúc]()
```

---

## 🚀 CÁCH CHẠY DỰ ÁN

```text
# 1️⃣ Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate        # Windows
# hoặc
source venv/bin/activate     # Linux / macOS

# 2️⃣ Cài đặt thư viện
pip install -r requirements.txt

# 3️⃣ Chạy demo
python Last.py
```
## 🔍 KẾT QUẢ HIỂN THỊ

- 🧾 **Khung người bị phát hiện (YOLOv8)**  
- 💬 **Xác suất gian lận (%)**  
- 📸 **Ảnh bằng chứng lưu tại:** `/logs/evidence/`  
- 🔔 **Âm thanh cảnh báo:** khi xác suất > ngưỡng  
- 🧠 **Mô tả hành vi từ BLIP-2:**  

**Ví dụ:**
> “Student using phone”  
> “Looking at another screen”  
> “Cheating detected”



