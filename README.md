# 🏥 Smart Biomedical Waste Segregation System (OR AI PoC)

An AI-powered Proof of Concept (PoC) designed for **Operating Room (OR) environments** to automatically classify biomedical waste into appropriate disposal categories using **YOLOv8 (Computer Vision)**.

---

## 🔍 Overview

Biomedical waste segregation is critical in hospitals to prevent infections, ensure safety, and comply with regulations.

This project uses:
- 🧠 Deep Learning (YOLOv8 Large Model)
- 📷 Real-time Camera Input (Webcam / DroidCam / IP Camera)
- 🎯 Object Detection + Custom Classification Logic
- 💡 LED Simulation for Bin Indication

---

## ♻️ Waste Categories

| Bin | Color | Type |
|-----|------|------|
| 🔴 Red | Infectious Waste | Contaminated items (gloves, gauze, blood samples) |
| 🟡 Yellow | Sharps Waste | Needles, blades, surgical tools |
| 🔵 Blue | Plastic/Glass/Pharma | Bottles, IV bags, vials |
| 🟢 Green | General Waste | Paper, packaging |

---

## ⚙️ Features

- ✅ Real-time object detection using **YOLOv8l**
- ✅ Multi-pass detection for higher accuracy
- ✅ Image preprocessing (CLAHE + denoising)
- ✅ Supports:
  - Webcam
  - DroidCam (Wi-Fi/USB)
  - IP Webcam
  - RTSP streams
- ✅ LED indicator simulation for bins
- ✅ Visual pipeline UI (step-by-step process)
- ✅ Automatic report generation
- ✅ Large surgical waste dataset mapping

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
