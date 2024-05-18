import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import math
import re
from sentence_transformers import SentenceTransformer
import torch
from numpy.linalg import norm
from pprint import pprint
import wikipedia
import sys
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

def get_first_sentence(text):
    pattern = re.compile(r'([.!?])\s+|([.!?])$')
    match = pattern.search(text)
    
    if match:
        end_index = match.end()
        return text[:end_index].strip()
    else:
        return text.strip()


def remove_top(df):
    winning_row = df.iloc[0]
    words = [winning_row['a_origin'], winning_row['b_origin'], winning_row['c_origin'], winning_row['d_origin']]
    words = set(words)
    df = df[~((df['a_origin'].isin(words)) & (df['b_origin'].isin(words)) & (df['c_origin'].isin(words)) & (df['d_origin'].isin(words)))]
    
    return df

summaries = []

def get_first_sentence(text):
    pattern = re.compile(r'([.!?])\s+|([.!?])$')
    match = pattern.search(text)
    
    if match:
        end_index = match.end()
        return text[:end_index].strip()
    else:
        return text.strip()


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
        summaries.append(get_first_sentence(summary))

wikipedia.set_lang("en") # Set language to English (or desired language)


def remove_words(specified_words, removals): 
    for word in removals:
        specified_words = [word1 for word1 in specified_words if word[0:word.index("_")] != word1[0:word1.index("_")]]
        
    return specified_words


device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

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
    global summaries
    
    start = time.time()
    
    for word in words:
        options = wikipedia.search(word.capitalize(), results=5)

        urls = [f'{base_url}{option.replace(" ", "_")}' for option in options]
        
        summaries = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(scrape_page, urls)
        for summary in summaries:
            wiki_dict["Word"].append(word)
            wiki_dict["Definition"].append(summary.strip())

    wiki_df = pd.DataFrame(wiki_dict)
    dict_df = pd.read_csv("connect/data/dictionary.csv")

    filtered_dict = pd.DataFrame(columns=dict_df.columns)
    
    for word in words:
        first_instance = dict_df[dict_df['Word'] == word].head(5)
        filtered_dict = pd.concat([filtered_dict, first_instance], ignore_index=True)
            
    dict_df = filtered_dict

    combined_words = pd.concat([wiki_df["Word"], dict_df["Word"]])
    combined_defs = pd.concat([wiki_df["Definition"], dict_df["Definition"]])

    wiki_df = pd.DataFrame({"Word": combined_words, "Definition": combined_defs})

    wiki_df['word_number'] = wiki_df.groupby('Word').cumcount() + 1
    wiki_df['Word'] = wiki_df.apply(lambda row: f"{row['Word']}_{row['word_number']}", axis=1)
    wiki_df = wiki_df.dropna()

    wiki_df["Definition"] = wiki_df["Definition"].astype(str)
    matrix = [retriever.encode(defi) for defi in wiki_df['Definition']]
    matrix = np.array(matrix)
    
    def cosine_similarity(a, b):
        return np.dot(a,b)/(norm(a)*norm(b))

    similarities = []
    
    prefixes = [a.split('_')[0] for a in wiki_df["Word"]]
    wiki_df = wiki_df.reset_index()
    words = wiki_df["Word"]

    for i in range(len(matrix)):
        a = matrix[i]
        for j in range(i + 1, len(matrix)):
            b = matrix[j]
            word1 = words[i]
            word2 = words[j]
            if prefixes[i] != prefixes[j]:
                sim = cosine_similarity(a, b)/math.dist(a, b)
                if math.isinf(sim):
                    sim = 1
                similarities.append([words[i], words[j], sim])
                
    df = pd.DataFrame(similarities, columns=["word_1", "word_2", "similarity"])
    
    # df = df[df["similarity"] > 0.03]

    relation_dict = {}

    for i, n in df.iterrows():
        word1 = n["word_1"]
        word2 = n["word_2"]
        
        key1 = (word1, word2)
        key2 = (word2, word1)
        
        relation_dict[key1] = n["similarity"]
        relation_dict[key2] = n["similarity"]


    specified_words = list(wiki_df["Word"])

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
        
        prefixes = [a.split('_')[0] for a in words]
        
        for i, a in enumerate(words):
            for j in range(i + 1, len(words)):
                if prefixes[j] == prefixes[i]:
                    continue
                b = words[j]
                for k in range(j + 1, len(words)):
                    if prefixes[k] in {prefixes[i], prefixes[j]}:
                        continue
                    c = words[k]
                    for l in range(k + 1, len(words)):
                        if prefixes[l] in {prefixes[i], prefixes[j], prefixes[k]}:
                            continue
                        d = words[l]
                        
                        df_dict_scores["a"].append(a)
                        df_dict_scores["a_origin"].append(prefixes[i])
                        df_dict_scores["b"].append(b)
                        df_dict_scores["b_origin"].append(prefixes[j])
                        df_dict_scores["c"].append(c)
                        df_dict_scores["c_origin"].append(prefixes[k])
                        df_dict_scores["d"].append(d)
                        df_dict_scores["d_origin"].append(prefixes[l])
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
    
    def remove_top(df):
        top = df.iloc[0]
        words = [top['a_origin'], top['b_origin'], top['c_origin'], top['d_origin']]
        words = set(words)
        df = df[~((df['a_origin'].isin(words)) & (df['b_origin'].isin(words)) & (df['c_origin'].isin(words)) & (df['d_origin'].isin(words)))]
        
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
            answers_df = remove_top(answers_df)

    if correct == 3:
        correct += 1
        tries += 1

        
    return {"correct": correct, "tries": tries}

connections = pd.read_csv("connect/data/connections2.csv")

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
        # sys.stdout = open(os.devnull, "w")
        # sys.stderr = open(os.devnull, "w")
        res = trial_connections(solutions=puzzle)
        # sys.stdout = old_stdout
        # sys.stderr = old_stderr
        end = time.time()
        
        func_time = end - start
        
        times.append(func_time)
        
        if res["correct"] == 4:
            puzzles.append(1)
        else:
            puzzles.append(0)
        
        connections_made.append(res["correct"])
        tries.append(res["tries"])
        
        print(f'Trial {int(i / 4) +  1}: Correct: {res["correct"]}\tTries: {res["tries"]}\tTime: {func_time}')
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