import re
import math
import random
import numpy as np
from utils import CharDataSet
import matplotlib.pyplot as plt

'''
N_tokens influences the way the fold are separeted. Initialize N_tokens similarly to the transformer model to obtain the same folds.
The variable N is the order of the N_gram : 2 or 3
'''

class N_gram:
    def __init__(self, N, N_tokens, dataset_path):
        self.dataset_path = dataset_path
        self.N_tokens = N_tokens
        self.N = N # order of the N_gram model
        self.epsilon = 1e-10 # probability for unseen characters
        self.decoder = {}
    
    def train_model(self, fold, k_fold) -> [float, float]: # returns perplexity_train, perplexity_validation
        tokenized_data_train_obj = CharDataSet(self.N_tokens, fold, k_fold, self.dataset_path, is_training=True)
        self.decoder = tokenized_data_train_obj.decoder # get the decoder dict to be able to convert numbers back to char
        tokenized_data_train = tokenized_data_train_obj.train_chunks # get the tokens
        tokenized_data_validation_obj = CharDataSet(self.N_tokens, fold, k_fold, self.dataset_path, is_training=False)
        tokenized_data_validation = tokenized_data_validation_obj.validation_chunks # get the tokens
        self.ngrams = {} # dict of word frequencies
        if self.N == 2 : # Bigram
            for i in range(len(tokenized_data_train)-1):
                w1 = tokenized_data_train[i].item()
                w2 = tokenized_data_train[i+1].item()
                if not w1 in self.ngrams:
                    self.ngrams[w1] = {}
                if not w2 in self.ngrams[w1]:
                    self.ngrams[w1][w2] = 1
                else:
                    self.ngrams[w1][w2] += 1 # increment count
        
            for c in self.ngrams: # go through keys
                self.ngrams[c] = dict(sorted(self.ngrams[c].items(), key=lambda item: -item[1])) # sort from most frequent
                total = sum(self.ngrams[c].values()) # sum all the count
                self.ngrams[c] = dict([(k, self.ngrams[c][k]/total) for k in self.ngrams[c]]) # compute relative frequency
        if self.N == 3:
            for i in range(len(tokenized_data_train)-2):
                w1 = tokenized_data_train[i].item()
                w2 = tokenized_data_train[i+1].item()
                w3 = tokenized_data_train[i+2].item()

                # Créez les dictionnaires nécessaires pour stocker les trigrammes
                if w1 not in self.ngrams:
                    self.ngrams[w1] = {}
                if w2 not in self.ngrams[w1]:
                    self.ngrams[w1][w2] = {}
                if w3 not in self.ngrams[w1][w2]:
                    self.ngrams[w1][w2][w3] = 1
                else:
                    self.ngrams[w1][w2][w3] += 1 # increment count
            
            # Traitez chaque niveau de la hiérarchie des dictionnaires
            for c1 in self.ngrams:
                for c2 in self.ngrams[c1]:
                    self.ngrams[c1][c2] = dict(sorted(self.ngrams[c1][c2].items(), key=lambda item: -item[1])) # sort from most frequent
                    total = sum(self.ngrams[c1][c2].values()) # sum all the count
                    self.ngrams[c1][c2] = dict([(k, self.ngrams[c1][c2][k]/total) for k in self.ngrams[c1][c2]]) # compute relative frequency
                    
        # Evaluate model
        perplexity_train = self.evaluate(tokenized_data_train) # eval on training dataset
        perplexity_validation = self.evaluate(tokenized_data_validation) #eval on validation dataset
        return perplexity_train, perplexity_validation
            
            
    def evaluate(self, tokenized_data) -> float: # training perplexity
        perplexity = 1 # neutral element for multiplication
        if self.N == 2:
            total_pairs = len(tokenized_data) - 1
            for i in range(total_pairs):
                current_word = tokenized_data[i].item()
                next_word = tokenized_data[i + 1].item()
                # Calculer la probabilité prédite P(w_{i+1} | w_i)
                try:
                    P_wi_plus_1_wi = self.ngrams[current_word][next_word]
                except: # caractère pas présent dans les données de training
                    P_wi_plus_1_wi = self.epsilon   
                # Calculer la perplexité pour cette paire de mots
                perplexity *= math.pow(1/P_wi_plus_1_wi,1/total_pairs)

        if self.N == 3:
            total_triples = len(tokenized_data) - 2
            for i in range(total_triples):
                current_word = tokenized_data[i].item()
                next_word = tokenized_data[i + 1].item()
                next_next_word = tokenized_data[i + 2].item()
                # Calculer la probabilité prédite P(w_{i+2} | w_{i+1}, w_i)
                try:
                    P_wi_plus_2_wi_plus_1_wi = self.ngrams[current_word][next_word][next_next_word]
                except: # suite de caractères pas présent dans les données de training
                    P_wi_plus_2_wi_plus_1_wi = self.epsilon
                
                # Calculer la log loss pour cette paire de mots
                perplexity *= math.pow(1/P_wi_plus_2_wi_plus_1_wi,1/total_triples)
        return perplexity
        
    def next_number(self, n1, n2 = 0): # returns next number generated
        '''
        Picks randomly the next number depending on the probability weights.
        Therefore the model does not just pick the most probable number for generation each time.
        '''
        if self.N == 2:
            vars = self.ngrams[n1]
            return random.choices(list(vars.keys()), weights=vars.values())[0]
        if self.N == 3:
            vars = self.ngrams[n1][n2]
            return random.choices(list(vars.keys()), weights=vars.values())[0]

    def generate_numbers(self, generation_size, start_nb_1, start_nb_2 = 0) -> list[int]:
        '''
        Function that generates a list of int representing the characters.
        '''
        if self.N == 2:
            nb_list = [start_nb_1]
            n1 = start_nb_1
            for i in range(generation_size):
                n2 = self.next_number(n1) # generate knowing n1
                nb_list.append(n2)
                n1 = n2
            return nb_list
        if self.N == 3:
            nb_list = [start_nb_1, start_nb_2]
            n1 = start_nb_1
            n2 = start_nb_2
            for i in range(generation_size):
                n3 = self.next_number(n1,n2) # generate knowing n1 and n2
                nb_list.append(n3)
                n1 = n2
                n2 = n3
            return nb_list


if __name__ == "__main__":
    
    # Init
    dataset = "./datasets/shakespear_corpus.txt"
    N_tokens = 64 # required to fold the dataset...
    bigram_model = N_gram(2,N_tokens,dataset)
    trigram_model = N_gram(3,N_tokens,dataset)
    
    # Single training 
    b_perplexity_train, b_perplexity_validation = bigram_model.train_model(1,10)
    t_perplexity_train, t_perplexity_validation = trigram_model.train_model(1,10)
        
    '''
    # Cross validation
    k_fold = 10
    b_perplexity_train_list = []
    b_perplexity_validation_list = []
    t_perplexity_train_list = []
    t_perplexity_validation_list = []
    for i in range(k_fold):
        b_perplexity_train, b_perplexity_validation = bigram_model.train_model(i,10)
        t_perplexity_train, t_perplexity_validation = trigram_model.train_model(i,10)
        b_perplexity_train_list.append(b_perplexity_train)
        b_perplexity_validation_list.append(b_perplexity_validation)
        t_perplexity_train_list.append(t_perplexity_train)
        t_perplexity_validation_list.append(t_perplexity_validation)
        
    print(b_perplexity_train_list)
    print(b_perplexity_validation_list)
    print(t_perplexity_train_list)
    print(t_perplexity_validation_list)
    '''
    
    # Generate numbers
    start_char_1 = 44 # start for bigram
    start_char_2 = 52 # start for trigram

    generation_size = 300
    
    generated_numbers_bigram = bigram_model.generate_numbers(generation_size, start_char_1)
    
    generated_numbers_trigram = trigram_model.generate_numbers(generation_size, start_char_1,start_char_2)
    
    #! Not working
    
    # Generate text
    
    # BUG : Comment faire pour reconvertir les nombres en caractères ? Les caractères ont été encodés par CharDataSet 
    
    generated_text_bigram = ""
    generated_text_trigram = ""

    generated_text_bigram = ''.join([bigram_model.decoder[nombre] for nombre in generated_numbers_bigram])

    generated_text_trigram = ''.join([trigram_model.decoder[nombre] for nombre in generated_numbers_trigram])
    
    print("Bigram text generation")
    print()
    print(generated_text_bigram)
    print()
    print("Trigram text generation")
    print()
    print(generated_text_trigram)
    