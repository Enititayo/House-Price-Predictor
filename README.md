# 🏠 House Price Prediction

A machine learning regression project that predicts house prices using XGBoost, with robust hyperparameter tuning via RandomizedSearchCV and RepeatedKFold cross-validation.

---

## 📁 Project Structure

```
├── data/
│   └── data.csv               # Raw housing dataset
├── models/
│   ├── h_model.pkl            # Trained XGBoost model
│   ├── x_test.pkl             # Saved test features
│   └── y_test.pkl             # Saved test labels
├── preprocess.py              # Feature engineering and data cleaning
├── train.py                   # Model training and hyperparameter tuning
├── evaluate.py                # Model evaluation and predictions
└── README.md
```

---

## ⚙️ How It Works

### 1. Preprocessing (`preprocess.py`)
Raw data is cleaned and transformed into meaningful features:

| Feature | Description |
|---|---|
| `apparent_age` | Weighted age of the house accounting for renovations |
| `quality` | Composite score from floors, waterfront, view, and condition |
| `sqft_vertical` | Total vertical space (above ground + basement) |
| `sqft_horizontal` | Total horizontal space (living area + lot) |
| `statezip` / `city` | Encoded as categorical features |

**Cleaning steps:**
- Removes listings with a price of 0
- Fills missing renovation years with the build year
- Drops irrelevant columns (`street`, `country`, `date`)

---

### 2. Training (`train.py`)
- Splits data 80/20 into train and test sets
- Log-transforms the target variable (`price`) to handle skewness
- Uses **RepeatedKFold** (5 splits × 5 repeats) for stable cross-validation
- Tunes hyperparameters with **RandomizedSearchCV** over 50 iterations
- Saves the best model and test data using `joblib`

**Tuned Parameters:**

| Parameter | Search Space |
|---|---|
| `n_estimators` | 100, 300, 500, 1000 |
| `max_depth` | 3, 6, 8, 10 |
| `learning_rate` | 0.3, 0.1, 0.05, 0.01 |
| `subsample` | 0.5, 0.6, 0.8, 1.0 |

---

### 3. Evaluation (`evaluate.py`)
- Loads the saved model and test data
- Reverses the log transform on predictions (`np.exp`)
- Reports **Mean Absolute Error (MAE)** and **R² Score**
- Prints a sample of actual vs predicted prices

**Current Performance:**
```
Mean Absolute Error: $97,633.55
R² Score:           0.70
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install xgboost scikit-learn pandas numpy joblib
```

### Run the Pipeline
```bash
# Step 1 - Train the model
python train.py

# Step 2 - Evaluate the model
python evaluate.py
```

> **Note:** Ensure your dataset is placed at `./data/data.csv` before running.

---

## 🛠️ Potential Improvements

- [ ] Add outlier removal to reduce the impact of extreme prices
- [ ] Expand hyperparameter search space (`colsample_bytree`, `gamma`, `reg_alpha`)
- [ ] Engineer additional features (`price_per_sqft`, `bed_bath_ratio`, `is_renovated`)
- [ ] Increase `n_iter` to 100 and `n_repeats` to 10 for more robust tuning
- [ ] Ensure `y_test` is also log-transformed for evaluation consistency

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `xgboost` | Gradient boosting regressor |
| `scikit-learn` | Preprocessing, model selection, metrics |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `joblib` | Model serialization |
