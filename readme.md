# 🤟 ASL Neural Interface (Real-Time Recognition)

A real-time American Sign Language (ASL) recognition system using **Python (Flask)** and **MediaPipe**. It features a server-side AI architecture for stability and a "Cyberpunk" styled frontend.

## 🚀 Features
* **Real-Time Detection:** Uses MediaPipe landmarks + Keras MLP.
* **Sentence Construction:** Typing logic with 'Space' and 'Delete' gestures.
* **Server-Side AI:** Robust backend processing (No WebGL crashes).
* **Reference Library:** Visual guide for all ASL signs.

## 🛠️ Setup & Installation

### 1. Prerequisites
* Python 3.8+
* Virtual Environment (Recommended)

### 2. Installation
```bash
# Clone the repo
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)

# Create & Activate Virtual Env
python -m venv my_env
source my_env/bin/activate  # Windows: my_env\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
