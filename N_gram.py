import re
import math
import random
import matplotlib.pyplot as plt


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
        self.epsilon = 1e-10 # probabilité pour les caractère jamais vu
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
            
    def evaluate_train(self) -> [float, float]:
        # Training evlauation
        total_loss = 0
        total_pairs = len(self.train_characters) - 1
        
        for i in range(total_pairs):
            current_word = self.train_characters[i]
            next_word = self.train_characters[i + 1]
            # Calculer la probabilité prédite P(w_{i+1} | w_i)
            try:
                P_wi_plus_1_wi = self.bgrams[current_word][next_word]
            except: # caractère pas présent dans les données de training
                P_wi_plus_1_wi = self.epsilon
            
            # Calculer la log loss pour cette paire de mots
            loss = -math.log(P_wi_plus_1_wi)
            total_loss += loss
        
        # Calculer la loss 
        loss = total_loss/len(self.train_characters)
        # Calculer la perplexité
        perplexity = math.pow(2,loss)
        return loss, perplexity
    
    def evaluate_validation(self) -> [float, float]:
        # Validation evaluation
        total_loss = 0
        total_pairs = len(self.validation_characters) - 1
        
        for i in range(total_pairs):
            current_word = self.validation_characters[i]
            next_word = self.validation_characters[i + 1]
            # Calculer la probabilité prédite P(w_{i+1} | w_i)
            try:
                P_wi_plus_1_wi = self.bgrams[current_word][next_word]
            except: # caractère pas présent dans les données de training
                P_wi_plus_1_wi = self.epsilon
            
            # Calculer la log loss pour cette paire de mots
            loss = -math.log(P_wi_plus_1_wi)
            total_loss += loss
        
        # Calculer la loss 
        loss = total_loss/len(self.validation_characters)
        # Calculer la perplexité
        perplexity = math.pow(2,loss)
        return loss, perplexity
    
    def save_losses_graph(self, path, losses):
        plt.clf()

        plt.plot(range(len(losses['train'])), losses['train'], label='Training_mean')
        plt.fill_between(range(len(losses['train'])), losses['train'] - losses['train_var'], losses['train'] + losses['train_var'], alpha=0.3, label='Variance Area for train')
        
        plt.plot(range(0, len(losses['train']), self.validation_interval), losses['validation'], label='Validation_mean')
        plt.fill_between(range(0, len(losses['train']), self.validation_interval), losses['validation'] - losses['validation_var'], losses['validation'] + losses['validation_var'], alpha=0.3, label='Variance Area for validation')
        plt.xlabel('Number of batches')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over number of batches')
        plt.legend()
        plt.savefig(path)
        plt.show()


    def save_perplexity_graph(self, path, perplexities):
        plt.clf()
        plt.plot(range(len(perplexities['train'])), perplexities['train'], label='Training_mean')
        plt.fill_between(range(len(perplexities['train'])), perplexities['train'] - perplexities['train_var'], perplexities['train'] + perplexities['train_var'], alpha=0.3, label='Variance Area for train')
        
        plt.plot(range(0, len(perplexities['train']), self.validation_interval), perplexities['validation'], label='Validation_mean')
        plt.fill_between(range(0, len(perplexities['train']), self.validation_interval), perplexities['validation'] - perplexities['validation_var'], perplexities['validation'] + perplexities['validation_var'], alpha=0.3, label='Variance Area for validation')
        plt.xlabel('Number of batches')
        plt.ylabel('Perplexity')
        plt.title('Training and Validation Perplexity Over number of batches')
        plt.legend()
        plt.savefig(path)
        plt.show()
        

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
