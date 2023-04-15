import pandas as pd

occurence_0=[]
file_name='/Users/genya/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
#file_name='/home/ipasichn/MerckActivity/TrainingSet/ACT{}_competition_training.csv'
for i in range(7,8):
  col_list=[]
  col_sel=[]
  fin_col=[]
  df=pd.read_csv(file_name.format(i))
  ncol=len(df.columns)
  for t in range(2, ncol):
    col_list=df[df.columns[t]].tolist()
    occurence_0.append(col_list.count(0))
    if col_list.count(0)==0:
      fin_col.append(t)
  print(fin_col)
  #col_sel.append(min(occurence_0))
  #print(col_sel, occurence_0.count(0))