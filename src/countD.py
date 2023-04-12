import pandas as pd

column_names=[]
prefix='D'
file_name='/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
for i in range(1,16):
  df=pd.read_csv(file_name.format(i), nrows=1)
  for col in df.columns:
    column_names.append(col)
descriptors=[x for x in column_names if x.startswith(prefix)]
unique_list=list(dict.fromkeys(descriptors))
print(unique_list, len(unique_list))