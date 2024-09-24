import pandas as pd
import numpy as np

file_path = input("Enter the file path that contains train and test data: ")

data = pd.read_csv(file_path + "train.csv")

## DATA CLEANING

# Removing Cabin as it has 77% missing values
data.drop('Cabin', axis=1, inplace=True)

# Replacing missing Age values with its mean as there were only 20% missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Embarked has 2% missing values, replacing it with its mode value
# mode is 'S' as it can be seen in the EDA report of embarked
data['Embarked'].fillna('S', inplace=True)

print(data.isna().sum())

# Removing 'Name' and 'Ticket' as it has very high cardinality
# 'Name' has 891 unique values, 'Ticket' has 681 unique values
# Removing these two as they don't have much predictive power
data.drop(['Name', 'Ticket'], axis=1, inplace=True)

## FEATURE ENGINEERING

# Converting 'Sex' into a numerical 0/1 value
data.replace({'male':1,'female':0}, inplace = True)

# Converting Fare to its log transformed value as it is highly right skewed
# Adding +1 to avoid -inf
data['Fare'] = np.log(data['Fare'] + 1)

## DATA CLEANING AND FEATURE ENGINEERING ON TEST SET

# Performing similar data cleaning and feature engineering on test data
test = pd.read_csv(file_path + 'test.csv')

test.drop('Cabin', axis=1, inplace=True)
test.drop(['Name', 'Ticket'], axis=1, inplace=True)

test['Age'].fillna(data['Age'].mean(), inplace=True)
test['Embarked'].fillna('S', inplace=True)

# Converting Embarked into one hot encoded numerical values as it has 3 categorical values
data = pd.get_dummies(data, columns=['Embarked'])

test.replace({'male':1,'female':0}, inplace = True)
test['Fare'] = np.log(test['Fare'] + 1)
test['Fare'].fillna(data['Fare'].mean(), inplace = True)
test = pd.get_dummies(test, columns=['Embarked'])

passenger_id_train = data['PassengerId'].copy()
passenger_id_test = test['PassengerId'].copy()

data.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)

data.to_csv(file_path + "train_processed.csv")
test.to_csv(file_path + "test_processed.csv")

print("Processed files are saved here: ", file_path)

