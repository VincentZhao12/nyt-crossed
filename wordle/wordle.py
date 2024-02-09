import math
from pprint import pprint
key = "penis"

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", mean = {0}):
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
    print(f'\r{prefix} |{bar}| {percent}% {suffix} Mean Count: {mean}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
class wordle_solver:
    def __init__(self):
        self.curr_word = [0, 0, 0, 0, 0]
        self.guess_num = 0
        self.letter_counts = {}
        file = open("words.txt", "r")
        self.words = file.readlines()

        self.words = [word.strip() for word in self.words if len(word.strip()) == 5]
    
    def make_guess(self):
        self.letter_counts = {}
        self.guess_num += 1
        
        for word in self.words:
            for i, ch in enumerate(word):
                if ch not in self.letter_counts:
                    self.letter_counts[ch] = [0, 0, 0, 0, 0]
                # self.letter_counts[ch][0] + 1
                self.letter_counts[ch][i] += 1
        # letter_dist = [[], [], [], [], []]
        # for i in range(5):
        #     for ch in self.letter_counts.keys():
        #         index = 0
        #         while index < len(letter_dist[i]) and letter_dist[i][index][1] > self.letter_counts[ch][0]:
        #             index += 1
        #         letter_dist[i].insert(index, (ch, self.letter_counts[ch][0]))
            
        # pprint(self.letter_counts)
        
        max_score = 0
        best_word = ""
        
        for word in self.words:
            score = self.score_word(word)
            
            if score > max_score:
                max_score = score
                best_word = word
        
        return best_word
        
    def score_word(self, word):
        score = 0
        
        curr_len = sum(self.curr_word)
        uniques = set()
        
        for i, ch in enumerate(word):
            uniques.add(ch)
            score += self.letter_counts[ch][i] + 0.2 * sum(self.letter_counts[ch])
            
        if (curr_len == 0 or curr_len == 1)and len(uniques) != 5:
            return 0
        return score 
    
    def guess_res(self, guess, correct_places, correct_letters):
        self.curr_word = correct_places
        guess_2 = guess
        
        for i, ch in enumerate(guess):
            if correct_places[i] == 1:
                self.words = [word for word in self.words if word[i] == ch]
                charlist = list(guess_2)
                charlist[i] = ' '
                guess_2 = ''.join(charlist)
        
        guess_3 = guess_2
        
        for i, ch in enumerate(guess_2):
            if correct_letters[i] == 1:
                self.words = [word for word in self.words if word[i] != ch and ch in word]
                charlist = list(guess_3)
                charlist[i] = ' '
                guess_3 = ''.join(charlist)
                
        for ch in guess_3:
            self.words = [word for word in self.words if not ch in word]
                
                
    
def check_guess(guess, key):
    correct_letters = [0, 0, 0, 0, 0]
    correct_places = [0, 0, 0, 0, 0]
    
    incorrects = set()
    
    for i, ch in enumerate(guess):
        if ch == key[i]:
            correct_places[i] = 1
        else:
            incorrects.add(ch)
    
    for i, ch in enumerate(guess):
        if ch in incorrects and ch in key and correct_places[i] == 0:
            correct_letters[i] = 1
            incorrects.remove(ch)
    
    return (correct_places, correct_letters)

def benchmark():
    file = open("words.txt", "r")
    all_words = file.readlines()

    all_words = [word.strip() for word in all_words if len(word.strip()) == 5]
    
    printProgressBar(0, len(all_words), prefix = 'Progress:', suffix = 'Complete', length = 50)
    # all_words = [all_words[0]]

    min_count = len(all_words)
    max_count = 0
    solved = 0
    counts = []

    for key in all_words:
        count = 0
        guess = ""
        solver = wordle_solver()
        # print("----------------------------------------")
        # print(key)

        while guess != key:
            guess = solver.make_guess()

            # print(guess)

            places, lets = check_guess(guess, key)
            
            # print(lets)
            # print(places)
            
            solver.guess_res(guess, places, lets)
            
            count += 1
            
            if guess == '':
                break;
            
        solved += 1
        max_count = max(count, max_count)
        min_count = min(count, min_count)
        counts.append(count)
        
        printProgressBar(solved, len(all_words), prefix = 'Progress:', suffix = 'Complete', length = 50, mean = sum(counts) / float(len(counts)))

    mean = sum(counts) / float(len(counts))

    std_dev = 0

    for n in counts:
        std_dev += (n - mean)**2
        
    std_dev = math.sqrt(std_dev / len (counts))

    print(f'max: {max_count} min: {min_count} mean: {mean} std dev: {std_dev}')
    
benchmark()