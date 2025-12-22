# 🚦 An Efficient Approach to Detect Helmet and Rider Liveliness Verification to Enhance Safety in EV

## 📌 Project Overview

This project presents a **real-time rider safety enforcement system** designed to ensure **helmet compliance** and **rider liveliness verification** for electric two-wheelers. The system integrates **YOLO-based helmet detection** and **face anti-spoofing detection**, supporting both **software-level testing** and **embedded hardware deployment** on a **Raspberry Pi 5**.

The repository is structured to clearly separate **software testing** and **hardware implementation**, enabling easier validation, debugging, and real-world deployment.

---

## 🧠 Core Functionalities

* Helmet detection using **YOLOv5n**
* Face anti-spoofing detection (**real vs spoof**)
* Dual-model integration with priority-based logic
* Confidence-based visual annotations
* LCD status display and buzzer alerts
* Motor control based on rider safety compliance

---

## 🧪 Project Modes

### 1️⃣ Software Mode (Testing & Validation)

This mode is intended for **model testing and validation** without hardware dependency.

**Purpose**

* Verify helmet detection accuracy
* Test face anti-spoofing logic
* Visualize bounding boxes and confidence scores

**Features**

* Runs on a PC/laptop
* Camera or video input
* On-screen visualization of detections

**Code Location**

```
codes/
 └── software_code.py
```

---

### 2️⃣ Hardware Mode (Embedded Deployment)

This mode implements the complete system on **Raspberry Pi 5** for real-world operation.

**Purpose**

* Real-time safety enforcement
* Embedded system validation
* EV safety integration

**Features**

* Pi Camera-based live video input
* YOLO helmet detection (local inference)
* Face anti-spoofing verification
* LCD display for real-time status
* Buzzer alerts for violations
* Motor control using a multi-stage state machine

**Code Location**

```
codes/
 └── hardware_code.py
```

---

## 🏗️ System Architecture

```
Pi Camera
   │
   ▼
Frame Capture (RGB)
   │
   ├── YOLOv5 Helmet Detection (Local)
   ├── Face Anti-Spoofing Model (API-based)
   │
   ▼
Decision Engine (State Machine)
   │
   ├── LCD Display
   ├── Buzzer Alerts
   └── Motor Control
```

---

## 📊 Dataset Information

* **Dataset Type**: Custom collected
* **Total Images**: ~2000
* **Classes**:

  * Helmet
  * No Helmet (includes masks, scarves, caps, burkas, etc.)
* **Annotation Format**: YOLO format
* ⚠️ **Dataset is not publicly released** due to privacy and academic constraints

---

## 🧠 Model Training & Results

### 🔹 Training Details

* Model: **YOLOv5n**
* Training Platform: **Google Colab**
* Dataset: Custom helmet dataset

### 🔹 Training Results 

The trained model demonstrated improved stability and accuracy compared to earlier versions, with reduced false positives and reliable real-time performance suitable for edge deployment.

*(Detailed metrics and graphs can be added in the `results/` directory if required.)*

---

## 🎥 Hardware Output Demonstration (Video)

The system output is demonstrated through a **recorded video**, showcasing:

* Live helmet detection
* Face anti-spoofing verification
* LCD status updates
* Buzzer alerts during violations
* Motor control behavior based on safety compliance

📌 **Demo Video**:
👉 *Add video link here (YouTube / Google Drive)/// will be added soon*

---

## 🖥️ Outputs & Alerts

* **Video Output**: Annotated bounding boxes with confidence scores
* **LCD Display**: Helmet and liveliness status
* **Buzzer**: Warning and violation alerts
* **Motor Control**: Speed regulation and shutdown

---

## ⚠️ Known Limitations

* Night vision not supported
* Performance affected by low-light conditions
* Dependency on external API for face anti-spoofing

---

## 🔮 Future Enhancements

* Fully offline face anti-spoofing model
* Infrared / night-vision camera support
* Cloud-based logging and analytics
* Rider identity verification integration

---

## 📂 Repository Structure

```
├── codes/
│   └── hardware_code.py
│   └── software_code.py
├── models/
│   └── Helmet_Detection.pt
├── results/
│   └── training_results.png
├── README.md
└── requirements.txt
```

---

## 🏁 Project Status

✔ Final-Year Major Project
✔ Hardware-validated prototype
✔ Real-world deployment oriented

---

## 📜 License

This project is intended for **academic and educational use only**.

---

## 👨‍💻 Author

**Guruprasad Kamath**
Electronics & Communication Engineering
Embedded AI | Computer Vision | Intelligent Systems


