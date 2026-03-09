#Import Libraries
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

#Load Models and Data
model = joblib.load("./models/h_model.pkl")
x_test = joblib.load("./models/x_test.pkl")
y_test = joblib.load("./models/y_test.pkl")

#Create Evaluation Function
def evaluate():
    #Predict Data
    y_pred = model.predict(x_test)
    y_pred = np.exp(y_pred)

    #Calaculate Error
    error = mean_absolute_error(y_test, y_pred)
    score = r2_score(y_test, y_pred)

    #Evaluation Outputs
    print("Actual\n")
    print(y_test[50:100])
    print("Predicted\n")
    print(y_pred[50:100])
    print(f"Mean Absolute Error: {round(error, 2)}")
    print(f"R2 Score: {round(score, 2)}")

#Execute Function
evaluate()