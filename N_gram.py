import re
import math
import random
import numpy as np
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
    
    def cross_validation(self, k_folds = 10):
        # Evaluation params
        loss_train_list = []
        perplexity_train_list = []
        loss_validation_list = []
        perplexity_validation_list = []
        
        fold_size = int(self.nb_characters / k_folds)
        for i in range(k_folds):
            validation_stard_index = i * fold_size
            validation_end_index = (i + 1) * fold_size
            # Border condition
            if((i+1) == k_folds ):
                validation_end_index = len(self.characters)-1 # final index of dataset
            self.train_characters = self.characters[0:validation_end_index] + self.characters[validation_end_index:]
            self.validation_characters = self.characters[validation_stard_index:validation_end_index]
            print("Validation indices : ", validation_stard_index, " to ", validation_end_index)
            print("train indices : 0 to ", validation_stard_index, " and ", validation_end_index, " to ", len(self.characters)-1)
            self.train()
            train_results = self.evaluate_train()
            validation_results = self.evaluate_validation()
            loss_train_list.append(train_results[0])
            perplexity_train_list.append(train_results[1])
            loss_validation_list.append(validation_results[0])
            perplexity_validation_list.append(validation_results[1])
        #print(loss_train_list)
        #print(perplexity_train_list)
        #print(loss_validation_list)
        #print(perplexity_validation_list)
        
        # Compute mean and var for train
        loss_train_mean = np.mean(np.array(loss_train_list))
        loss_train_var = np.var(np.array(loss_train_list))
        perplexity_train_mean = np.mean(np.array(perplexity_train_list))
        perplexity_train_var = np.var(np.array(perplexity_train_list))
        
        # Compute mean and var for validation
        loss_validation_mean = np.mean(np.array(loss_validation_list))
        loss_validation_var = np.var(np.array(loss_validation_list))
        perplexity_validation_mean = np.mean(np.array(perplexity_validation_list))
        perplexity_validation_var = np.var(np.array(perplexity_validation_list))
        
        return loss_train_mean, loss_train_var, perplexity_train_mean, perplexity_train_var, loss_validation_mean, loss_validation_var, perplexity_validation_mean, perplexity_validation_var

        

    def train(self):
        self.bgrams = {}
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
            
    def evaluate_train(self) -> [float, float]: # loss, perplexity
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
    
    def evaluate_validation(self) -> [float, float]: # loss, perplexity
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
