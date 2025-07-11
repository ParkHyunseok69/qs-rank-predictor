# 🎓 QS University Rank Predictor

Predict your university's 2026 QS World Rank using real metrics — built with Machine Learning, XGBoost, and Streamlit.

---

## 📌 Overview

This project uses a **stacked regression model** (Linear Regression + Random Forest + XGBoost) trained on the **2026 QS World University Rankings dataset** to estimate a university’s QS Rank based on key performance indicators.

🔮 **Features:**
- Predicts QS Rank based on academic and internationalization scores
- Explains most influential metrics
- Benchmark against other universities in your region or country
- Save & load custom input profiles for repeated use
- Clean, responsive Streamlit UI

---

## 🚀 Live App

https://qs-rank-predictor.streamlit.app/

---

## 📊 Model Details

- Preprocessing: OneHotEncoding + StandardScaler
- Feature Selection: RandomForestRegressor + SelectFromModel
- Final Model: StackingRegressor with Linear & Forest base models, XGBoost as meta-learner
- Metrics Used: 13 total including `AR SCORE`, `ER SCORE`, `Overall SCORE`, `Previous Rank`, etc.

📈 **Model Performance**

R² Score: 0.9918
MAE: ~26
RMSE: ~46

---

## ⚙️ How to Use Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/qs-rank-predictor.git
cd qs-rank-predictor
pip install -r requirements.txt
streamlit run app.py

├── app.py                # Streamlit frontend
├── model.py              # ML pipeline and model training
├── uni_rank_predictor.pkl # Trained pipeline model
├── requirements.txt      # Project dependencies
├── 2026 QS World University Rankings.csv  # Dataset
└── README.md
