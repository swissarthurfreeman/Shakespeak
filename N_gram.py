import re
import math
import random


class BigramLanguageModel:
    def __init__(self, dataset):
        self.dataset = dataset
        self.CharDataSet()
        
    def CharDataSet(self):
        self.data = self.load_data(self.dataset)
        self.characters = list(self.data) # split text to char list
        self.nb_characters = len(self.characters)
        self.train_characters = self.characters[:int(0.9*self.nb_characters)]
        self.validation_characters = self.characters[int(0.9*self.nb_characters):]
        self.bgrams = {}
        
    def get_vocab(self):
        return set(self.characters)

    def train(self):
        for i in range(len(self.train_characters)-2):
            w1 = self.train_characters[i]
            w2 = self.train_characters[i+1]
            if not w1 in self.bgrams:
                self.bgrams[w1] = {}
            if not w2 in self.bgrams[w1]:
                self.bgrams[w1][w2] = 1
            else:
                self.bgrams[w1][w2] += 1
    
        for c in self.bgrams: # go through keys
            self.bgrams[c] = dict(sorted(self.bgrams[c].items(), key=lambda item: -item[1])) # sort from most frequent
            total = sum(self.bgrams[c].values()) # sum all the count
            self.bgrams[c] = dict([(k, self.bgrams[c][k]/total) for k in self.bgrams[c]]) # compute relative frequency
            
    def evaluate(self):
        # Training evlauation
        total_loss = 0
        total_pairs = len(self.train_characters) - 1
        
        for i in range(total_pairs):
            current_word = self.train_characters[i]
            next_word = self.train_characters[i + 1]
            # Calculer la probabilité prédite P(w_{i+1} | w_i)
            P_wi_plus_1_wi = self.bgrams[current_word][next_word]
            
            # Calculer la log loss pour cette paire de mots
            loss = math.log(1/P_wi_plus_1_wi)
            total_loss += loss
        
        # Calculer la perplexité logarithmique sur l'ensemble d'entraînement
        perplexity_log = math.exp(total_loss / len(self.train_characters))
        # Calculer la perplexité 
        perplexity = math.exp(perplexity_log)
        return perplexity
        # Validation evaluation
        

    def next_char(self, char):
        vars = self.bgrams[char]  
        return random.choices(list(vars.keys()), weights=vars.values())[0]

    def generate_text(self, start_char):
        chars = []
        c = start_char
        for i in range(50):
            c = self.next_char(c)
            if c == '.':
                break
            chars.append(c)
        return (' '.join(chars) + '. ').capitalize()

    def load_data(self, filename) -> str:
        with open(filename, 'r') as file:
            data = file.read() # read data
            #data = data.lower().replace("\n", ' ') # lower case
            #data = re.sub('[^a-z0-9\.\-]', ' ', data) # remove non-alpha_numeric
            #data = re.sub('\s{2,}', ' ', data) # remove multiple spaces
            #data = '. ' + data.replace('.', ' .') # 
            #data = re.sub(' [a-z0-9] ', '', data) # remove single char (reduce noise)
        return data
