import pandas as pd
import os
from gensim.models import KeyedVectors
from scipy.stats import pearsonr
import numpy as np
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.test.utils import datapath
def read_data(path):
    data = pd.read_csv(path,sep='\t',header=0)

    return data

#Loading SimLex-999

path = os.getcwd()
data = read_data('SimLex-999.txt')

output1 = open("output1.txt",'w')

#Printing 1.2
print("Printing SimLex999 similarity...",file=output1)
s1=data["SimLex999"][(data["word1"]=="hard") & (data["word2"]=="easy")].values[0]
s2=data["SimLex999"][(data["word1"]=="hard") & (data["word2"]=="difficult")].values[0]
s3=data["SimLex999"][(data["word1"]=="hard") & (data["word2"]=="dense")].values[0]
print("(hard, easy): ",s1,file = output1)
print("(hard, difficult): ",s2,file = output1)
print("(hard, dense): ",s3,file = output1)
print("Completed!\n",file=output1)

#Loading word2vec word embeddings
wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=100000)



def get_word2vec(wv,w1,w2):
    v1 = wv[w1] if w1 in wv.vocab else np.zeros(wv.vector_size)
    v2 = wv[w2] if w2 in wv.vocab else np.zeros(wv.vector_size)
    return v1,v2

def compute_euclidean_distance(mv,w1,w2):
    v1,v2 = get_word2vec(wv,w1,w2)
    distance = np. linalg. norm(v1 - v2)
    return distance

#Printing 1.3
d1=compute_euclidean_distance(wv, "head", "easy")
d2=compute_euclidean_distance(wv, "head", "difficult")
d3=compute_euclidean_distance(wv, "head", "dense")
print("Computing euclidean distance...", file = output1)
print("(head, easy): ",d1,file = output1)
print("(head, difficult): ",d2,file = output1)
print("(head, dense): ",d3,file = output1)
print("Completed!\n",file=output1)


def compute_pearson(data,wv):
    similarities = data["SimLex999"]
    distances = []
    for row in data.itertuples():
        distances.append(compute_euclidean_distance(wv,row[0],row[1]))
    return pearsonr(similarities, distances)

print("Computing Pearson's correlation coefficient...",file=output1)
print("Pearson's correlation coefficient is ",compute_pearson(data,wv)[0],file=output1)
output1.close()
#print(word_vectors.shape())
#def compute_word2vec(pairs):
 #   #v1,v2 = word2vec.word2vec(pairs)







