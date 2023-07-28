import pandas as pd

row_names=[]
prefix='ACT'
#file_name='/Users/genya/projects/MerckActivity/TestSet/ACT{}_competition_test.csv'
file_name = '/home/ewgeni/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
for i in range(1,16):
  df=pd.read_csv(file_name.format(i), usecols=[0])
  row_names=df['MOLECULE'].tolist()
  #row_names=row_names+df['MOLECULE'].tolist()
  #print(row_names)
  #molecules=[x for x in row_names if x.startswith(prefix)]
  #unique_list=list(dict.fromkeys(molecules))
#print(unique_list, len(unique_list))
  #print(len(row_names))
  unique_list=list(dict.fromkeys(row_names))
print(unique_list, len(unique_list))