# 💳 Credit Card Fraud Detection using XGBoost

\
A high-performance credit card fraud detection model trained on real anonymized transactional data, achieving an **AUC score of 0.978** using **XGBoost** and **SMOTE** for handling data imbalance.

---

## 🚀 Project Highlights

- 🔍 **Model**: XGBoost Classifier (optimized for imbalanced data)
- 📊 **AUC Score**: 0.978 (Excellent detection capability)
- 📈 **Feature Engineering**: PCA-transformed V1–V28, plus Time & Amount
- ⚖️ **Class Imbalance**: Handled using SMOTE oversampling
- 📉 **Evaluation**: ROC Curve, Confusion Matrix, Classification Report
- 🧠 **Top Features**: Extracted from XGBoost's gain-based importance
- ✅ **Custom Transaction Prediction** supported
- 🌐 **Interactive Web App**: Streamlit-based user interface for real-time fraud detection predictions

---

## 📁 Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Rows**: 284,807 transactions
- **Frauds**: Only \~0.17% → highly imbalanced
- **Features**:
  - V1–V28: Principal Components (PCA)
  - Time: Seconds elapsed between transactions
  - Amount: Transaction amount
  - Class: `1` = Fraud, `0` = Not Fraud

---

## 🧠 Technologies Used

| Tool                 | Purpose                     |
| -------------------- | --------------------------- |
| Python 3.9+          | Programming Language        |
| XGBoost              | Gradient Boosted Classifier |
| SMOTE                | Oversampling for imbalance  |
| Scikit-Learn         | Preprocessing & Evaluation  |
| Pandas & NumPy       | Data handling               |
| Matplotlib & Seaborn | Visualizations              |

---

## 🧪 How It Works

1. **Load and explore data**
2. **Scale numeric features**
3. **Split dataset into training/testing sets**
4. **Balance data using SMOTE**
5. **Train XGBoost with fine-tuned hyperparameters**
6. **Evaluate using classification metrics + ROC AUC**
7. **Identify top features using model importance**
8. **Interactive fraud prediction using Streamlit app**

---

## 📈 Model Performance

| Metric            | Score    |
| ----------------- | -------- |
| AUC-ROC           | 0.978    |
| Precision (Fraud) | High     |
| Recall (Fraud)    | High     |
| False Positives   | Very Low |

### 🔍 Sample Prediction

```python
Prediction: Legit ✅
Fraud Probability: 0.00186
```

---

## 📊 Visuals (ROC Curve & Top Features)

- **ROC Curve:**\
- ![image](https://github.com/user-attachments/assets/494b9847-1faa-46cc-94c6-de10f1fd7e73)



- **Top 10 Features:**\
- ![image](https://github.com/user-attachments/assets/5b1cf5f0-9d00-43b3-b761-dccb3ccadb3a)



---

## 🧪 Example Use: Predict Fraud on Custom Transaction

```python
new_tx = np.array([[...30 scaled values...]])
scaled_tx = scaler.transform(new_tx)
model.predict(scaled_tx)
```

---

## 📦 Project Structure

```
├── creditcard.csv
├── credit_card_fraud_detection.ipynb
├── README.md
├── fraud_model.pkl  (optional)
├── scaler.pkl  (optional)
├── streamlit_app.py  (Streamlit app for real-time fraud prediction)
├── assets/
│   ├── roc_curve.png
│   └── feature_importance.png
```

---

## 🛠️ Future Improvements

- Hyperparameter tuning with GridSearchCV
- Real-time deployment via Streamlit/Gradio
- Automated threshold selection
- Integration with APIs or transaction systems

---

## 📜 License

MIT License – use freely with attribution.

