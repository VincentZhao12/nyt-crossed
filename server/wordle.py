import math

class wordle_solver:
    def __init__(self):
        self.curr_word = [0, 0, 0, 0, 0]
        self.guess_num = 0
        file = open("data/words.txt", "r")
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
                    word_rep = letter_counts[ch][1]
                    word_rep[i] += 1
                    letter_counts[ch] = (letter_counts[ch][0] + 1, word_rep)
        
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
        
        diff_letters = set()
        
        for i, ch in enumerate(word):
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