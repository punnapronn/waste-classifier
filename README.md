# Waste Classifier (MobileNetV2 Transfer Learning)

โปรเจกต์นี้ใช้พลังของ **Deep Learning (Convolutional Neural Network - CNN)**  ร่วมกับเทคนิค **Transfer Learning จาก MobileNetV2**  เพื่อสร้างระบบจำแนกประเภทขยะที่สามารถบอกได้ว่า **ภาพนี้คือขยะประเภทใด** เช่น Plastic, Paper, Metal, Glass, Organic, Battery ฯลฯ 

## Download Dataset

[click](https://www.kaggle.com/datasets/wasifmahmood01/custom-waste-classification-dataset/data) for install Dataset from Kaggle

## Install Virtual Environment

สร้าง Virtual Environment
```bash
python -m venv waste-cnn
```
เปิดใช้งาน venv (windows)
```bash
waste-cnn\Scripts\activate
```
ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```
## train model
```python
python src/train.py
```
เมื่อเทรนเสร็จ ไฟล์โมเดลจะถูกบันทึกไว้ที่ ```models/best_model.h5``` และจะเก็บ history ไว้ที่ ```outputs/history.json```

## Evaluate
```python
python src/evaluate.py
```
ผลลัพธ์ที่ได้:
- Confusion Matrix → ```outputs/figures/confusion_matrix.png```
- รายงานสรุป (precision, recall, f1-score) → ```outputs/report.json```
- ค่า accuracy และ summary แสดงใน Terminal

## ตั้งค่าการ train model
ไฟล์ ```src/config.py``` เก็บพารามิเตอร์หลัก เช่น:
```python
EPOCHS = 50
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
```
## Streamlit App
รันแอปเว็บเพื่อทดสอบโมเดลแบบ interactive:
```bash
streamlit run app/streamlit_app.py
```
แอปจะเปิดหน้าเว็บ (เช่น http://localhost:8501)
สามารถอัปโหลดภาพขยะแล้วดูผลการจำแนกได้ทันที