import pandas as pd

file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
for i in range(1,2):
  df=pd.read_csv(file_name.format(i), header=None, skiprows=1, nrows=2)
  print(df)