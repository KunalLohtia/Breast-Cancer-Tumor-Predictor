import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import classification_report
import pickle

# Load preprocessed breast cancer dataset
data = pd.read_csv("breast-cancer.csv")
data = data.drop("id", axis = 1)

# deleting the rows with any instance of 0 in it, assume they are not real data
data = data[(data['concavity_mean'] != 0 )]

# Assuming your target variable is in a column named 'target'
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train.values.ravel())

# Make predictions on the test set
y_pred = model.predict(X_test)

# make pickle file of the model, mlp also used to fit data so have to pass mlp as the object
pickle.dump(model, open("logistic.pkl", "wb"))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", round(accuracy, 4))

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Repeat the process for multiple epochs
num_runs = 6
average_accuracy = 0

for _ in range(num_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()

    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)

    average_accuracy += accuracy_score(y_test, y_pred)

average_accuracy /= num_runs

print("\nAverage accuracy (over", num_runs, "epochs): ", round(average_accuracy, 4))

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Input values for prediction
input_data = np.array([12.18,20.52,77.22,458.7,0.08013,0.04038,0.02383,0.0177,0.1739,0.05677,0.1924,1.571,1.183,14.68,0.00508,0.006098,0.01069,0.006797,0.01447,0.001532,13.34,32.84,84.58,547.8,0.1123,0.08862,0.1145,0.07431,0.2694,0.06878]).reshape(1, -1)

# Standardize the input data using the previously fit scaler
input_data_standardized = scaler.transform(input_data)

# Make predictions
prediction = model.predict(input_data_standardized)

# Output the prediction
print("Predicted Diagnosis:", prediction[0])
