# 🏠 Real Estate Price Prediction using XGBoost

This repository contains a trained **XGBoost Regression model** for predicting real estate property prices based on structured housing features. The model has been trained, tuned, and exported as a `model.json` file for easy reuse and deployment.

---

## 📌 Project Overview

The goal of this project is to build a robust machine learning model that can accurately predict property prices using key real estate attributes such as location, size, property type, and legal status.

- **Model Type:** XGBoost Regressor
- **Problem Type:** Supervised Regression
- **Framework:** XGBoost (scikit-learn API)
- **Saved Model Format:** `model.json`

---

## 📂 Repository Structure

```
├── train.csv              # Training dataset
├── test.csv               # Test dataset
├── reg_model.json         # Trained XGBoost model
├── README.md              # Project documentation
```

---

## 🧾 Features Used

The model was trained using the following features:

| Feature Name | Description |
|-------------|------------|
| POSTED_BY | Property posted by (encoded) |
| UNDER_CONSTRUCTION | Whether property is under construction |
| RERA | RERA registration status |
| BHK_NO. | Number of bedrooms |
| BHK_OR_RK | BHK or RK type |
| SQUARE_FT | Total area in square feet |
| READY_TO_MOVE | Ready-to-move status |
| RESALE | Resale or new property |
| ADDRESS | Encoded address/location |
| LONGITUDE | Longitude of property |
| LATITUDE | Latitude of property |

⚠️ **Important:** All categorical features were converted to numerical values before training. The same encoding must be applied during inference.

---

## ⚙️ Model Hyperparameters

The model was trained using the following optimized hyperparameters:

```python
XGBRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1.0
)
```

These settings help balance **bias–variance tradeoff** and reduce overfitting.

---

## 🚀 How to Load the Trained Model

You can load and use the trained model directly using XGBoost:

```python
from xgboost import XGBRegressor

model = XGBRegressor()
model.load_model("reg_model.json")
```

---

## 📈 Making Predictions

Ensure the input data:
- Has the **same feature order** as training
- Uses the **same preprocessing and encoding**

```python
y_pred = model.predict(X_test)
```

---

## 🔁 Preprocessing Notes

- Categorical features were **numerically encoded before training**
- No raw string categories are supported at inference time
- Feature scaling was not mandatory due to tree-based model usage

📌 **Best Practice:** Save preprocessing steps (encoders, imputers) separately using `joblib` for production use.

---

## 🧠 Key Learnings

- XGBoost performs well on tabular real estate data
- Proper regularization is critical for large boosting rounds
- Feature consistency is mandatory when loading saved models

---

## 📌 Future Improvements

- Native categorical handling using `enable_categorical=True`
- Hyperparameter tuning with Optuna
- Model explainability using SHAP
- Deployment via FastAPI or Streamlit

---

## 👤 Author

**Nisarg Patel**  
Aspiring Data Scientist | Machine Learning Enthusiast

---

⭐ If you find this project useful, consider giving it a star!

