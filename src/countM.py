import pandas as pd

row_names=[]

file_name = '/home/ewgeni/projects/MerckActivity/TrainingSet/ACT{}_competition_training.csv'

for i in range(1,16):
  df = pd.read_csv(file_name.format(i), usecols=[0])

  tmpstr = 'ACT' + str(i) + '_'
  df['MOLECULE'] = df['MOLECULE'].str.replace(tmpstr, '')

  #row_names = df['MOLECULE'].tolist()
  row_names = row_names + df['MOLECULE'].tolist()

  #molecules=[x for x in row_names if x.startswith(prefix)]
  #unique_list=list(dict.fromkeys(molecules))
#print(unique_list, len(unique_list))
  #print(len(row_names))
unique_list=list(dict.fromkeys(row_names))
import collections
print([item for item, count in collections.Counter(row_names).items() if count > 1])
#print(row_names, len(row_names), type(row_names))
#print(unique_list, len(unique_list))