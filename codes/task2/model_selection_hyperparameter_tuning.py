import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flaml import AutoML
import logging
import joblib

file_path = input("Enter the processed data location here: ")

data = pd.read_csv(file_path + "train_processed.csv")

scaler = StandardScaler()

X = data.drop('Survived', axis=1)
y = data['Survived']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

automl = AutoML()

# Define settings for the AutoML search
automl_settings = {
    "time_budget": 60,  # Time alloted to get the best model. Intially it was set to 5 mins, but it took a long time to execute. So changed it to 1 to proceed with the assignment
                        # It has to be changed to get better results
    "metric": 'accuracy',
    "task": 'classification',
    "estimator_list": ['rf', 'xgboost', 'lgbm'],
    "log_file_name": 'titanic_flaml.log',  # Generating the Log file
    "seed": 42,
}

# Run AutoML to search for the best model
automl.fit(X_train=X_train_scaled, y_train=y_train, **automl_settings)

# Get the best model
print("Best ML model:", automl.best_estimator)
best_model = automl.model

print("----------------------------------")
print("Best model: ", best_model)

best_hyperparameters = automl.best_config
print("Best hyperparameters:", best_hyperparameters)

joblib.dump(best_model, '/content/model/best_model.joblib')
joblib.dump(scaler, "/content/model/StandardScaler.joblib")

print("Best model and standard scaler are stored as joblib files")

y_pred = automl.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")