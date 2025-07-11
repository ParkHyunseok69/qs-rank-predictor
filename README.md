Absolutely! Here's the updated `README.md` with an **Author** section and the **MIT license badge** added at the top for a clean, professional look:

---

````markdown
# 🎓 QS University Rank Predictor

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A machine learning-powered Streamlit web app that predicts a university’s **2026 QS World University Rank** based on official QS indicators like Academic Reputation, Faculty-Student Ratio, International Faculty Ratio, and more.

---

## 🚀 Features

- ✅ **Rank Prediction** using StackingRegressor (Linear + Forest → XGBoost)
- 🧮 **Key Metric Inputs**: AR SCORE, CPF SCORE, Overall SCORE, etc.
- 🧠 **Model Explanation**: Highlights the most influential features
- 💾 **Save & Load Profiles**: For comparing different metric combinations
- 🌍 **Benchmarking**: Shows how your scores compare to others in your country/region
- 📊 **Interactive UI** built with Streamlit

---

## 📂 How to Run

1. Clone the repo:

```bash
git clone https://github.com/ParkHyunseok69/qs-rank-predictor.git
cd qs-rank-predictor
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## 🧠 Model Architecture

* Preprocessing: `StandardScaler`, `OneHotEncoder`
* Feature Selection: `SelectFromModel` using `RandomForestRegressor`
* Base Models: `LinearRegression`, `RandomForestRegressor`
* Meta Learner: `XGBoostRegressor`
* Final MAE: \~26.7 | RMSE: \~46 | R²: 0.99+

---

## 📘 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Lawrence Andre' Q. Cabana**
📫 [Visit my GitHub](https://github.com/ParkHyunseok69)
🔭 I’m building AI tools with practical real-world value.
🧠 Always learning — currently into ML, game dev, and automation.

---

> *This app is for educational and demonstration purposes only.*

```

---

Let me know if you'd also like to:
- Add a contributors section
- Add deployment instructions (for Streamlit Cloud)
- Add a demo video or GIF preview at the top  
```
