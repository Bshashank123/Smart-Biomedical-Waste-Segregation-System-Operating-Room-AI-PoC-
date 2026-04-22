# 🏥 Smart Biomedical Waste Segregation System (OR AI PoC)

An AI-powered Proof of Concept (PoC) designed for **Operating Room (OR) environments** to automatically classify biomedical waste into appropriate disposal categories using **YOLOv8 (Computer Vision)**.

---

## 🔍 Overview

Biomedical waste segregation is critical in hospitals to prevent infections, ensure safety, and comply with regulations.

This project uses:

* 🧠 Deep Learning (YOLOv8 Large Model)
* 📷 Real-time Camera Input (Webcam / DroidCam / IP Camera)
* 🎯 Object Detection + Custom Classification Logic
* 💡 LED Simulation for Bin Indication

---

## ♻️ Waste Categories

| Bin       | Color                | Type                                              |
| --------- | -------------------- | ------------------------------------------------- |
| 🔴 Red    | Infectious Waste     | Contaminated items (gloves, gauze, blood samples) |
| 🟡 Yellow | Sharps Waste         | Needles, blades, surgical tools                   |
| 🔵 Blue   | Plastic/Glass/Pharma | Bottles, IV bags, vials                           |
| 🟢 Green  | General Waste        | Paper, packaging                                  |

---

## ⚙️ Features

* ✅ Real-time object detection using **YOLOv8l**
* ✅ Multi-pass detection for higher accuracy
* ✅ Image preprocessing (CLAHE + denoising)
* ✅ Supports:

  * Webcam
  * DroidCam (Wi-Fi/USB)
  * IP Webcam
  * RTSP streams
* ✅ LED indicator simulation for bins
* ✅ Visual pipeline UI (step-by-step process)
* ✅ Automatic report generation
* ✅ Large surgical waste dataset mapping

---

## 🧠 How It Works

1. Capture frame from camera
2. Enhance image (lighting + noise reduction)
3. Run YOLO detection (multi-pass)
4. Map detected objects → waste categories
5. Highlight object with bounding boxes
6. Activate corresponding bin (LED simulation)
7. Generate classification report

---

## 🛠️ Installation

```bash
pip install ultralytics opencv-python numpy
```

---

## ▶️ Usage

### 📷 1. Webcam (default)

```bash
python smart_waste_poc.py
```

### 📱 2. DroidCam / IP Webcam

```bash
python smart_waste_poc.py --droidcam 192.168.1.42
```

### 🌐 3. RTSP Stream

```bash
python smart_waste_poc.py --droidcam rtsp://192.168.1.42:4747/h264_pcm.sdp
```

### 🖼️ 4. Image Input

```bash
python smart_waste_poc.py --image sample.jpg
```

---

## 🎛️ Parameters

| Argument  | Description                          |
| --------- | ------------------------------------ |
| `--conf`  | Detection confidence (default: 0.30) |
| `--iou`   | IOU threshold (default: 0.45)        |
| `--model` | YOLO model (default: yolov8l.pt)     |

---

## 📊 Output

* 📁 Captured images stored in `/output`
* 🖼️ Result images with bounding boxes
* 📋 Terminal classification report
* 💡 LED indicator simulation (UI)

---

## 🧩 Tech Stack

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* NumPy

---

## 🏥 Real-World Applications

* Hospital Operating Rooms
* Automated Waste Disposal Systems
* Smart Healthcare Infrastructure
* Infection Control Systems

---


## 🚀 Future Improvements

* 🔌 Arduino / ESP32 integration for real LED bins
* 🧠 Custom-trained medical dataset
* ☁️ Cloud logging & dashboard
* 📱 Mobile app interface
* 🔊 Voice feedback system

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---


## 👨‍💻 Author

Shashank Bodduna
