import pandas as pd
import os
def read_data(path):
    data = pd.read_table(path,sep='\t',header=0)

    return data


path = os.getcwd()
data = read_data('SimLex-999.txt')

output1 = open("output1.txt",'w')

print("(hard, easy): ",data["SimLex999"][(data["word1"]=="hard") & (data["word2"]=="easy")].values[0],
      file = output1)
print("(hard, difficult): ",data["SimLex999"][(data["word1"]=="hard") & (data["word2"]=="difficult")].values[0],
      file = output1)
print("(hard, dense): ",data["SimLex999"][(data["word1"]=="hard") & (data["word2"]=="dense")].values[0],
      file = output1)


