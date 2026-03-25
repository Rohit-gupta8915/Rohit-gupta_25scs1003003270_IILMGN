# Rohit-gupta_25scs1003003270_IILMGN
# 🌍 AI-Based Land Use Recommendation & Prediction System

## 📌 Overview
This project is an AI-driven system that predicts and recommends optimal land usage based on environmental and demographic factors.

It combines:
- Machine Learning (Random Forest)
- Rule-Based Reasoning
- GUI (Tkinter)

The system suggests land usage categories like:
- 🌾 Agriculture
- 🏙 Urban Development
- 🌳 Forest Conservation
- 🔄 Mixed Use

---

## 🎯 Problem Statement
Traditional land-use planning is:
- Slow
- Manual
- Inconsistent

This project solves that by automating land-use decisions using AI.

---

## 🧠 Key Features

✔ Machine Learning Prediction (Random Forest)  
✔ Rule-Based Explanation System  
✔ Conflict Detection (ML vs Rules)  
✔ User-Friendly GUI  
✔ Real-time Predictions (< 0.5 sec)  
✔ Confidence Score Output  

---

## 🛠 Tech Stack

- **Language:** Python  
- **ML Library:** Scikit-learn  
- **GUI:** Tkinter  
- **Data Handling:** Pandas, NumPy  

---

## 📊 Input Parameters

The model takes:

- Population Density  
- Soil Fertility  
- Area Type  
- Latitude  
- Longitude  

---

## ⚙️ How It Works

1. Input land data via GUI  
2. ML model predicts land-use category  
3. Rule-based system explains the decision  
4. Conflict detection checks inconsistencies  
5. Output shows recommendation + confidence  

---

## 🧪 Model Details

- Algorithm: Random Forest Classifier  
- Trees: 200  
- Accuracy: **82% – 90%**  

---
---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn
python land_use_gui_with_model.py
