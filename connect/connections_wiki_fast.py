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
import sys
import time

base_url = "https://en.wikipedia.org/wiki/"

wikipedia.set_lang("en")

summaries = []

def cosine_similarity(a, b):
    return np.dot(a,b)/(norm(a)*norm(b))

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


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
# device = torch.device("mps")

retriever = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device = device
)

print("sentence embedder loaded")

base_url = "https://en.wikipedia.org/wiki/"


def trial_connections(solutions):
    words = []
    
    for row in solutions:
        for word in row:
            words.append(word)
            
    wiki_dict = {"Word": [], "Definition": []}
    
    for word in words:
        options = wikipedia.search(word.capitalize(), results=10)

        urls = [f'{base_url}{option.replace(" ", "_")}' for option in options]
        
        summaries = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(scrape_page, urls)
        for summary in summaries:
            wiki_dict["Word"].append(word)
            wiki_dict["Definition"].append(summary.strip())

    wiki_df = pd.DataFrame(wiki_dict)
    
    dict_df = pd.read_csv("connect/data/dictionary.csv")


    dict_df["Word"] = dict_df["Word"].str.upper()
    dict_df = dict_df[dict_df["Word"].isin(words)]
    dict_df = dict_df.reset_index()

    combined_words = pd.concat([wiki_df["Word"], dict_df["Word"]])
    combined_defs = pd.concat([wiki_df["Definition"], dict_df["Definition"]])

    wiki_df = pd.DataFrame({"Word": combined_words, "Definition": combined_defs})

    wiki_df["Definition"] = wiki_df["Definition"].astype(str)
    
    embeddings = [retriever.encode(defi) for defi in wiki_df['Definition']]
    embeddings = np.array(embeddings)

    embedding_size = embeddings.shape[1]
    
    embedding_dict = {}

    for word in words:
        embedding_dict[word] = np.zeros(embedding_size)

    for i, word in enumerate(wiki_df["Word"]):
        embedding_dict[word] += embeddings[i]

    embedding_dict
    

    similarities = []

    for i in range(len(words)):
        a = embedding_dict[words[i]]
        for j in range(i + 1, len(words)):
            b = embedding_dict[words[j]]
            word1 = wiki_df.iloc[i]["Word"]
            word2 = wiki_df.iloc[j]["Word"]
            
            sim = cosine_similarity(a, b) / (math.dist(a, b))
            if math.isinf(sim):
                sim = 1
            similarities.append([words[i], words[j], sim])
                
    df = pd.DataFrame(similarities, columns=["word_1", "word_2", "similarity"])

    df.sort_values("similarity")


    relation_dict = {}

    for i, n in df.iterrows():
        word1 = n["word_1"]
        word2 = n["word_2"]
        
        key1 = (word1, word2)
        key2 = (word2, word1)
        
        relation_dict[key1] = n["similarity"]
        relation_dict[key2] = n["similarity"]


    # @lru_cache(maxsize=None)
    def similarity_4(a, b, c, d):
        return relation_dict[(a, b)] + relation_dict[(a, c)] + relation_dict[(a, d)] + relation_dict[(b, c)] + relation_dict[(b, d)] + relation_dict[(c, d)]


    def find_groups(words):
        df_dict_scores = {
            'a': [],
            'a_origin': [],
            'b': [],
            'b_origin': [],
            'c': [],
            'c_origin': [],
            'd': [],
            'd_origin': [],
            'sim': [],
        }
        for i, a in enumerate(words):
            for j in range(i + 1, len(words)):
                b = words[j]
                for k in range(j + 1, len(words)):
                    c = words[k]
                    for l in range(k + 1, len(words)):
                        d = words[l]
                        
                        df_dict_scores["a"].append(a)
                        df_dict_scores["a_origin"].append(a.split('_')[0])
                        df_dict_scores["b"].append(b)
                        df_dict_scores["b_origin"].append(b.split('_')[0])
                        df_dict_scores["c"].append(c)
                        df_dict_scores["c_origin"].append(c.split('_')[0])
                        df_dict_scores["d"].append(d)
                        df_dict_scores["d_origin"].append(d.split('_')[0])
                        df_dict_scores["sim"].append(similarity_4(a, b, c, d))
        
        return pd.DataFrame.from_dict(df_dict_scores)
    
    def not_one_away(df):
        winning_row = df.iloc[0]
        words = [winning_row['a_origin'], winning_row['b_origin'], winning_row['c_origin'], winning_row['d_origin']]
        words = set(words)
        df = df[~((df['a_origin'].isin(words)) & (df['b_origin'].isin(words)) & (df['c_origin'].isin(words)))]
        df = df[~((df['b_origin'].isin(words)) & (df['c_origin'].isin(words)) & (df['d_origin'].isin(words)))]
        df = df[~((df['c_origin'].isin(words)) & (df['d_origin'].isin(words)) & (df['a_origin'].isin(words)))]
        df = df[~((df['d_origin'].isin(words)) & (df['a_origin'].isin(words)) & (df['b_origin'].isin(words)))]
        
        return df

    def check_win(df):
        row = df.iloc[0]
        
        for sol in solutions:
            solution_set = set(sol)
            if (row['a_origin'] in solution_set) and (row['b_origin'] in solution_set) and (row['c_origin'] in solution_set) and (row['d_origin'] in solution_set):
                return True
            
        return False
        
    def check_one_away(df):
        row = df.iloc[0]
        
        for sol in solutions:
            solution_set = set(sol)
            if (row['a_origin'] in solution_set) and (row['b_origin'] in solution_set) and (row['c_origin'] in solution_set):
                return True
            if (row['b_origin'] in solution_set) and (row['c_origin'] in solution_set) and (row['d_origin'] in solution_set):
                return True
            if (row['c_origin'] in solution_set) and (row['d_origin'] in solution_set) and (row['a_origin'] in solution_set):
                return True
            if (row['d_origin'] in solution_set) and (row['a_origin'] in solution_set) and (row['b_origin'] in solution_set):
                return True
            
        return False
        
    def after_win(df):
        winning_row = df.iloc[0]
        words = [winning_row['a_origin'], winning_row['b_origin'], winning_row['c_origin'], winning_row['d_origin']]
        words = set(words)
        df = df[~((df['a_origin'].isin(words)) | (df['b_origin'].isin(words)) | (df['c_origin'].isin(words)) | (df['d_origin'].isin(words)))]
        
        return df
    
    answers_df = find_groups(words)
    answers_df = answers_df.sort_values("sim", ascending=False)
    
    tries = 0
    correct = 0
    
    while (tries - correct) < 4 and correct < 3:
        tries += 1
        if check_win(answers_df):
            answers_df = after_win(answers_df)
            correct += 1
        elif not check_one_away(answers_df):
            answers_df = not_one_away(answers_df)
        else:
            answers_df = answers_df.iloc[1:, :]

    if correct == 3:
        correct += 1
        tries += 1

        
    return {"correct": correct, "tries": tries}

connections = pd.read_csv("connect/data/connections.csv")

times = []
connections_made = []
tries = []
puzzles = []

old_stdout = sys.stdout
old_stderr = sys.stderr

for i in range(0, len(connections), 4):
    if len(connections) - i < 4:
        break
    
    puzzle = []
    for j in range(4):
        row = connections.iloc[i + j]
        
        puzzle.append([row["word1"], row["word2"], row["word3"], row["word4"]])
        
    try:
        start = time.time()
        
        res = trial_connections(solutions=puzzle)
        
        end = time.time()
        
        func_time = end - start
        
        times.append(func_time)
        
        if res["correct"] == 4:
            puzzles.append(1)
        else:
            puzzles.append(0)
        
        connections_made.append(res["correct"])
        tries.append(res["tries"])
        
        print(f'Trial {int(i / 4 + 1)}: Correct: {res["correct"]}\tTries: {res["tries"]}\tTime: {func_time}')
    except KeyboardInterrupt as e:
        sys.stdout = old_stdout
        break;
    except Exception as e:
        sys.stdout = old_stdout
        print(e)
        print(puzzle)
    
print(f'On a set of {len(puzzles)} puzzles')
print(f'Puzzles Solved: {sum(puzzles)}\tConnections Made: {sum(connections_made)}\tTries Made: {sum(tries)}\tConnection Find Rate: {float(sum(connections_made)) / sum(tries)}')
print(f'Average Time Per Puzzle: {float(sum(times)) / len(times)}s\tBenchmarking took {sum(times)}s')
print(times)
print(puzzles)
print(connections_made)
print(tries)