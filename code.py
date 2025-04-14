# ---------------------------------------
# ✅ STEP 1: Install necessary packages
# ---------------------------------------
!pip install xgboost imbalanced-learn

# ---------------------------------------
# ✅ STEP 2: Import libraries
# ---------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ---------------------------------------
# ✅ STEP 3: Load data
# ---------------------------------------
df = pd.read_csv('creditcard.csv')
print("Data Loaded ✅ Shape:", df.shape)

# Class distribution visualization
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()

# ---------------------------------------
# ✅ STEP 4: Preprocessing
# ---------------------------------------
X = df.drop(['Class'], axis=1)
y = df['Class']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# ---------------------------------------
# ✅ STEP 5: Handle imbalance with SMOTE
# ---------------------------------------
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print("After SMOTE ✅:", np.bincount(y_res))

# ---------------------------------------
# ✅ STEP 6: Train XGBoost
# ---------------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_res, y_res)
print("Model trained ✅")

# ---------------------------------------
# ✅ STEP 7: Make predictions
# ---------------------------------------
y_pred = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

# ---------------------------------------
# ✅ STEP 8: Evaluation metrics
# ---------------------------------------
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("📈 ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label="XGBoost (AUC = {:.4f})".format(roc_auc_score(y_test, y_pred_prob)))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------
# ✅ STEP 9: Feature Importance
# ---------------------------------------
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Important Features")
plt.show()

# ---------------------------------------
# ✅ STEP 10: Output Prediction Example
# ---------------------------------------
i = 7  # Change this index to test other transactions
print(f"\n🔍 Prediction for transaction {i}:")
print("Actual Class        :", y_test.values[i])
print("Predicted Class     :", y_pred[i])
print("Fraud Probability   :", y_pred_prob[i])

# ---------------------------------------
# ✅ STEP 11: Predict on custom input
# ---------------------------------------
# Replace values with real inputs (must be 30 features)
custom_tx = X_test[5].reshape(1, -1)  # sample from test set

custom_pred = xgb_model.predict(custom_tx)[0]
custom_prob = xgb_model.predict_proba(custom_tx)[0][1]

print("\n💡 Custom Transaction Prediction:")
print("Predicted Class     :", "FRAUD 🚨" if custom_pred == 1 else "Legit ✅")
print("Fraud Probability   :", custom_prob)

# ---------------------------------------
# ✅ STEP 12: View Result Table
# ---------------------------------------
result_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Fraud_Probability': y_pred_prob
})

# Show first 10 predictions
print("\n📋 Top 10 Predictions:")
print(result_df.head(10))
