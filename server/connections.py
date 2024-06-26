import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import math
from sentence_transformers import SentenceTransformer
import torch
from numpy.linalg import norm
import wikipedia
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

base_url = "https://en.wikipedia.org/wiki/"
relation_dict = {}

wikipedia.set_lang("en")

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

retriever = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device = device
)

base_url = "https://en.wikipedia.org/wiki/"

summaries = []

def cosine_similarity(a, b):
    norms = (norm(a)*norm(b))
    dot = np.dot(a,b)
    if norms == 0:
        return 0
    return dot / norms

def scrape_page(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    

    paragraphs = soup.find('div', {'id': 'mw-content-text'}).find_all('p')

    summary_element = None
    for paragraph in paragraphs:
        if paragraph.get_text(strip=True): 
            summary_element = paragraph
            break

    summary = summary_element.get_text().strip()

    if len(summary) > 5:
        summaries.append(summary)

def remove_words(specified_words, removals): 
    for word in removals:
        specified_words = [word1 for word1 in specified_words if word[0:word.index("_")] != word1[0:word1.index("_")]]
        
    return specified_words

def make_wiki_df(words):
    wiki_dict = {"Word": [], "Definition": []}
    
    global summaries
    
    for word in words:
        options = wikipedia.search(word.capitalize(), results=10)

        urls = [f'{base_url}{option.replace(" ", "_")}' for option in options]
        
        summaries = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(scrape_page, urls)
        for summary in summaries:
            wiki_dict["Word"].append(word)
            wiki_dict["Definition"].append(summary.strip())

    return pd.DataFrame(wiki_dict)

@lru_cache(maxsize=None)
def similarity_4(a, b, c, d):
    return relation_dict[(a, b)] + relation_dict[(a, c)] + relation_dict[(a, d)] + relation_dict[(b, c)] + relation_dict[(b, d)] + relation_dict[(c, d)]

def find_connections(words):
    wiki_df = make_wiki_df(words)
    
    # dict_df = pd.read_csv("data/dictionary.csv")

    # dict_df["Word"] = dict_df["Word"].str.upper()
    # dict_df = dict_df[dict_df["Word"].isin(words)]
    # dict_df = dict_df.reset_index()

    # combined_words = pd.concat([wiki_df["Word"], dict_df["Word"]])
    # combined_defs = pd.concat([wiki_df["Definition"], dict_df["Definition"]])

    # wiki_df = pd.DataFrame({"Word": combined_words, "Definition": combined_defs})

    wiki_df["Definition"] = wiki_df["Definition"].astype(str)
    
    embeddings = [retriever.encode(defi) for defi in wiki_df['Definition']]
    embeddings = np.array(embeddings)

    embedding_size = embeddings.shape[1]
    
    embedding_dict = {}

    for word in words:
        embedding_dict[word] = np.zeros(embedding_size)

    for i, word in enumerate(wiki_df["Word"]):
        embedding_dict[word] += embeddings[i]
    
    similarities = []

    for i in range(len(words)):
        a = embedding_dict[words[i]]
        for j in range(i + 1, len(words)):
            b = embedding_dict[words[j]]
            word1 = wiki_df.iloc[i]["Word"]
            word2 = wiki_df.iloc[j]["Word"]
            
            sim = cosine_similarity(a, b)
            dist = (math.dist(a, b))
            
            if dist != 0:
                sim /= dist

            if math.isinf(sim):
                sim = 1
            similarities.append([words[i], words[j], sim])
                
    df = pd.DataFrame(similarities, columns=["word_1", "word_2", "similarity"])

    df.sort_values("similarity")

    for i, n in df.iterrows():
        word1 = n["word_1"]
        word2 = n["word_2"]
        
        key1 = (word1, word2)
        key2 = (word2, word1)
        
        relation_dict[key1] = n["similarity"]
        relation_dict[key2] = n["similarity"]

    df_dict_scores = []
    
    for i, a in enumerate(words): 
        for j in range(i + 1, len(words)):
            b = words[j]
            for k in range(j + 1, len(words)):
                c = words[k]
                for l in range(k + 1, len(words)):
                    d = words[l]
                    
                    df_dict_scores.append({
                        "words": [a, b, c, d],
                        "similarity": similarity_4(a, b, c, d)
                    })
                    
    return sorted(df_dict_scores, key=lambda x: x["similarity"], reverse=True)