import math
key = "penis"

class wordle_solver:
    def __init__(self):
        self.curr_word = [0, 0, 0, 0, 0]
        self.guess_num = 0
        file = open("words.txt", "r")
        self.words = file.readlines()

        self.words = [word[0:len(word) - 1].strip() for word in self.words]
    
    def make_guess(self):
        letter_counts = {}
        self.guess_num += 1
        
        for word in self.words:
            for i, ch in enumerate(word):
                if self.curr_word[i] != 1:
                    if ch not in letter_counts:
                        letter_counts[ch] = (0, [0, 0, 0, 0, 0])
                    # letter_counts[ch][0] + 1
                    word_rep = letter_counts[ch][1]
                    word_rep[i] += 1
                    letter_counts[ch] = (letter_counts[ch][0] + 1, word_rep)
                    
        # most_freq = []
        
        # for key in letter_counts.keys():
        #     if len(most_freq) == 0:
        #         most_freq.append((key, letter_counts[key][0]))
        #     for i, n in enumerate(most_freq):
        #         if letter_counts[key][0] > n[1]:
        #             most_freq.insert(i, (key, letter_counts[key][0]))
        #             break
        
        max_score = 0
        best_word = ""
        
        for word in self.words:
            score = self.score_word(word, letter_counts)
            
            if score > max_score:
                max_score = score
                best_word = word
        
        return best_word
        
    def score_word(self, word, letter_counts):
        score = 0
        
        # num_correct = sum(self.curr_word)
        diff_letters = set()
        
        for i, ch in enumerate(word):
            # score += letter_counts[ch][0] + letter_counts[ch][1][i]
            if self.curr_word[i] != 1:
                score += 0.1 * letter_counts[ch][0] + letter_counts[ch][1][i]
                diff_letters.add(ch)
            
        return score + len(self.words) * 0.2 * len(diff_letters)
    
    def guess_res(self, guess, correct_places, correct_letters):
        self.curr_word = correct_places
        
        for i, ch in enumerate(guess):
            if correct_places[i] == 1:
                self.words = [word for word in self.words if word[i] == ch]
            elif correct_letters[i] == 1:
                self.words = [word for word in self.words if word[i] != ch and ch in word]
            else:
                self.words = [word for word in self.words if not ch in word]
                
                
    
def check_guess(guess):
    correct_letters = [0, 0, 0, 0, 0]
    correct_places = [0, 0, 0, 0, 0]
    
    for i, ch in enumerate(guess):
        if ch == key[i]:
            correct_places[i] = 1
        elif ch in key:
            correct_letters[i] = 1
    
    return (correct_places, correct_letters)

file = open("words.txt", "r")
all_words = file.readlines()

all_words = [word[0:len(word) - 1].strip() for word in all_words]

min_count = len(all_words)
max_count = 0
counts = []

for key in all_words:
# if True:
    count = 0
    guess = ""
    solver = wordle_solver()

    while guess != key:
        guess = solver.make_guess()

        # print(guess)

        places, lets = check_guess(guess)
        solver.guess_res(guess, places, lets)

        # print(lets)
        # print(places)
        
        count += 1
    max_count = max(count, max_count)
    min_count = min(count, min_count)
    counts.append(count)

mean = sum(counts) / float(len(counts))

std_dev = 0

for n in counts:
    std_dev += (n - mean)**2
    
std_dev = math.sqrt(std_dev / len (counts))

print(f'max: {max_count} min: {min_count} mean: {mean} std dev: {std_dev}')