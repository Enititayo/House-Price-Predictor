#Import Libraries
from preprocess import streamline_features
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import numpy as np

#Clean and Transform Data
X, Y = streamline_features("./data/data.csv")

#Split Data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, train_size=0.8)
y_train = np.log(y_train)

#Create and Train Model
model = XGBRegressor(n_estimators=1000, learning_rate=0.009, subsample=0.5, enable_categorical=True, max_depth=5)
model.fit(x_train, y_train)

#Save Model and Values
joblib.dump(model, "./models/h_model.pkl")
joblib.dump(x_test, "./models/x_test.pkl")
joblib.dump(y_test, "./models/y_test.pkl")

#Print Average R2 Score
scores = cross_val_score(model, X, Y, cv=10, scoring="r2")
print(scores)
print(f"Average R2 Score: {round(scores.mean(), 2)}")