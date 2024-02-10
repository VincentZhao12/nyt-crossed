#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import math
import re
from sentence_transformers import SentenceTransformer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint


# In[27]:


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

retriever = SentenceTransformer(
    "paraphrase-MiniLM-L6-v2",
    device = device
)


# In[28]:


dict_df = pd.read_csv("data/dictionary.csv")

dict_df


# In[29]:


words = [
    "ACHE", "BURN", "SMART", "STING",
    "GUARD", "MIND", "TEND", "WATCH",
    "FINE", "PRIME", "QUALITY", "STERLING",
    "CAVITY", "CROWN", "FILLING", "PLAQUE"
]


# In[30]:


dict_df["Word"] = dict_df["Word"].str.upper()


# In[31]:


dict_df = dict_df[dict_df["Word"].isin(words)]

dict_df = dict_df.reset_index()
dict_df


# In[32]:


embeddings = retriever.encode(dict_df['Definition'])

embeddings.shape


# In[33]:


matrix = embeddings


# In[34]:


def cosine_similarity(a, b):
    return np.dot(a,b)/(norm(a)*norm(b))


# In[45]:


similarities = []

for i in range(len(matrix)):
    a = matrix[i]
    for j in range(i, len(matrix)):
        b = matrix[j]
        if dict_df.iloc[i]["Word"] != dict_df.iloc[j]["Word"]:
            similarities.append([dict_df.iloc[i]["Word"], dict_df.iloc[j]["Word"], cosine_similarity(a, b)/math.dist(a, b)])
            
df = pd.DataFrame(similarities, columns=["word_1", "word_2", "similarity"])

df


# In[46]:


df = df.groupby(['word_1', 'word_2'])['similarity'].max().reset_index()

df


# In[48]:


df = df.drop_duplicates()
df = df.dropna()
df = df.sort_values(by="similarity", ascending=False)

df


# In[49]:


relation_dict = {}

for i, n in df.iterrows():
    word1 = n["word_1"]
    word2 = n["word_2"]
    
    key1 = (word1, word2)
    key2 = (word2, word1)
    
    relation_dict[key1] = n["similarity"]
    relation_dict[key2] = n["similarity"]

relation_dict


# In[50]:


sim_4 = []

def similarity_4(a, b, c, d):
    return relation_dict[(a, b)] + relation_dict[(a, c)] + relation_dict[(a, d)] + relation_dict[(b, c)] + relation_dict[(b, d)] + relation_dict[(c, d)]

for i, a in enumerate(words):
    for j in range(i + 1, len(words)):
        b = words[j]
        for k in range(j + 1, len(words)):
            c = words[k]
            for l in range(k + 1, len(words)):
                d = words[l]
                try:
                    score = similarity_4(a, b, c, d)
                        
                    index = 0
                    
                    while index < len(sim_4) and score > sim_4[index][1]:
                        index += 1
                    sim_4.insert(index, ([a, b, c, d], score))
                except:
                    pass
                    

pprint(sim_4)

