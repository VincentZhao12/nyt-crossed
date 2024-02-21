import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import math
import re
from sentence_transformers import SentenceTransformer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import norm
import heapq
from pprint import pprint
import wikipedia
import sys
import os
import time

wikipedia.set_lang("en") # Set language to English (or desired language)


def remove_words(specified_words, removals): 
    for word in removals:
        specified_words = [word1 for word1 in specified_words if word[0:word.index("_")] != word1[0:word1.index("_")]]
        
    return specified_words


def cosine_similarity(a, b):
    return np.dot(a,b)/(norm(a)*norm(b))

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

retriever = SentenceTransformer(
    "paraphrase-MiniLM-L6-v2",
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
    
    soups = [BeautifulSoup(requests.get(f'{base_url}{word.capitalize()}').content, 'html.parser') for word in words]

    for i, soup in enumerate(soups):
        content = soup.find("div", id="mw-content-text")

        first_para = content.find("p").text

        items = [item.text for item in content.find_all("li")]

        if f'may refer to:' in first_para:
            for item in items:
                item = item.lower()
                if words[i] in item:
                    item = item.sub(words[i].lower(), "")

                if len(item) > 5:
                    wiki_dict["Word"].append(words[i])
                    wiki_dict["Definition"].append("".join(item))


    for word in words:
        options = wikipedia.search(word.capitalize(), results=10)
        for option in options:
            try:
                summary = wikipedia.summary(option, sentences=1, auto_suggest=False)
                # summary = page.summary
                if len(summary) > 10:
                    wiki_dict["Word"].append(word)
                    wiki_dict["Definition"].append(summary)
            except wikipedia.exceptions.DisambiguationError as e:
                pass
    wiki_df = pd.DataFrame(wiki_dict)
    dict_df = pd.read_csv("connect/data/dictionary.csv")


    dict_df["Word"] = dict_df["Word"].str.upper()
    dict_df = dict_df[dict_df["Word"].isin(words)]
    dict_df = dict_df.reset_index()

    combined_words = pd.concat([wiki_df["Word"], dict_df["Word"]])
    combined_defs = pd.concat([wiki_df["Definition"], dict_df["Definition"]])

    wiki_df = pd.DataFrame({"Word": combined_words, "Definition": combined_defs})

    wiki_df['word_number'] = wiki_df.groupby('Word').cumcount() + 1
    wiki_df['Word'] = wiki_df.apply(lambda row: f"{row['Word']}_{row['word_number']}", axis=1)
    wiki_df = wiki_df.dropna()


    wiki_df["Definition"] = wiki_df["Definition"].astype(str)
    matrix = [retriever.encode(defi) for defi in wiki_df['Definition']]
    matrix = np.array(matrix)
    
    print('data loaded')

    similarities = []

    for i in range(len(matrix)):
        a = matrix[i]
        for j in range(i, len(matrix)):
            b = matrix[j]
            word1 = wiki_df.iloc[i]["Word"]
            word2 = wiki_df.iloc[j]["Word"]
            if word1[0: word1.index("_")] != word2[0: word2.index("_")]:
                sim = cosine_similarity(a, b)/math.dist(a, b)
                if math.isinf(sim):
                    sim = 1
                similarities.append([wiki_df.iloc[i]["Word"], wiki_df.iloc[j]["Word"], sim])
                
    df = pd.DataFrame(similarities, columns=["word_1", "word_2", "similarity"])

    df = df[df["similarity"] > 0.03]


    relation_dict = {}

    for i, n in df.iterrows():
        word1 = n["word_1"]
        word2 = n["word_2"]
        
        key1 = (word1, word2)
        key2 = (word2, word1)
        
        relation_dict[key1] = n["similarity"]
        relation_dict[key2] = n["similarity"]


    specified_words = list(wiki_df["Word"])

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
            printProgressBar(iteration=i, total=len(words))
            for j in range(i + 1, len(words)):
                b = words[j]
                if a[0:a.index("_")] == b[0:b.index("_")]:
                    continue
                if (a, b) not in relation_dict:
                    continue
                for k in range(j + 1, len(words)):
                    c = words[k]
                    if a[0:a.index("_")] == c[0:c.index("_")] or b[0:b.index("_")] == c[0:c.index("_")]:
                        continue
                    if (a, c) not in relation_dict or (b, c) not in relation_dict:
                        continue
                    for l in range(k + 1, len(words)):
                        d = specified_words[l]
                        
                        if a[0:a.index("_")] == d[0:d.index("_")] or b[0:b.index("_")] == d[0:d.index("_")] or c[0:c.index("_")] == d[0:d.index("_")]:
                            continue
                        if ((a, d) not in relation_dict) or ((b, d) not in relation_dict) or ((c, d) not in relation_dict):
                            continue
                        
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
        print(words)
        df = df[~((df['a_origin'].isin(words)) | (df['b_origin'].isin(words)) | (df['c_origin'].isin(words)) | (df['d_origin'].isin(words)))]
        
        return df
    
    answers_df = find_groups(specified_words)
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
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        res = trial_connections(solutions=puzzle)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        end = time.time()
        
        func_time = end - start
        
        times.append(func_time)
        
        if res["correct"] == 4:
            puzzles.append(1)
        else:
            puzzles.append(0)
        
        connections_made.append(res["correct"])
        tries.append(res["tries"])
        
        print(f'Correct: {res["correct"]}\tTries: {res["tries"]}\tTime: {func_time}')
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