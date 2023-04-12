import pandas as pd

merge=pd.DataFrame()
file_name='/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
for i in range(1,3):
  df=pd.read_csv(file_name.format(i), header=None, skiprows=1, nrows=2)
  print(df)