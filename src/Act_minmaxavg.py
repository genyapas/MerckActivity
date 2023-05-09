import pandas as pd

row_names=[]
file_name='/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
for i in range(7,8):
  df=pd.read_csv(file_name.format(i))
  row_names=df['Act'].tolist()
  #row_names=row_names+df['MOLECULE'].tolist()
  #print(row_names)
  #molecules=[x for x in row_names if x.startswith(prefix)]
  #unique_list=list(dict.fromkeys(molecules))
#print(unique_list, len(unique_list))
  #print(len(row_names))
  print(min(row_names), max(row_names), sum(row_names)/len(row_names))
  unique_list=list(dict.fromkeys(row_names))
#print(unique_list, len(unique_list))