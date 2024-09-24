from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime import submodular_pick
import lime.lime_tabular
import numpy as np
import lightgbm as lgb
import time
import shap
import warnings

warnings.filterwarnings("ignore")

file_path = input("Enter the model stored location: ")

lgb_model = joblib.load(file_path + 'best_model.joblib')
scaler = joblib.load(file_path + 'StandardScaler.joblib')

data = pd.read_csv("/content/data/train_processed.csv")

X = data.drop(['Survived'], axis = 1).copy()
y = data['Survived'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

feature_names = X_train.columns.tolist()

explainer = LimeTabularExplainer(X_train_scaled
                                 , training_labels = y_train.to_numpy()
                                 , mode='classification'
                                 , feature_names= [column for column in X_train.columns if column != 'Survived']
                                 , categorical_features = [0,1,3,4,6,7,8]
                                 , class_names = [0, 1]
                                 )

samples = random.choices(list(range(0, len(X_test_scaled))), k = 5)
print("LIME is calculated on the following samples randomly chosen from test set:", samples)

X_test_samples = X_test_scaled[samples,:]
y_test_samples = y_test.to_numpy()[samples]

# Model output probability required to get LIME explanations
def return_prob(data):
    p1,p2 = (np.ones(len(data))[0] - lgb_model.predict(data)),  lgb_model.predict(data)
    prediction = [[x,y] for x,y in zip(p1,p2)]
    return lgb_model.predict_proba(data)

# Make predictions using the best model
y_pred = lgb_model.predict(X_test)  # For predicted classes
y_prob = lgb_model.predict_proba(X_test)  # For predicted probabilities

print("Predicted classes for the random test samples:", y_pred[:5])  # First 5 predictions
print("Predicted probabilities for the random test samples:", y_prob[:5])  # First 5 probabilities

print("------------------------------")
print("LIME results:")
for index, sample in enumerate(X_test_samples):
  print('Sample:', index + 1)
  print('Actual output:', y_test_samples[index])
  exp = explainer.explain_instance(sample, return_prob, top_labels = 2, num_features=9)
  print(exp.as_list())
  fig = exp.as_pyplot_figure()
  plt.savefig("/content/data/plot"+str(index)+".png")
  plt.show()
