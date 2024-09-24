import pandas as pd
import numpy as np
from dataprep.eda import create_report

file_path = input("Enter the train csv file path: ")
save_path = input("Enter the EDA report path: ")

data = pd.read_csv(file_path)

report = create_report(data)

report.save(save_path + "titanic_EDA.html")