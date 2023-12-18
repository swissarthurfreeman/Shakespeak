import math
import torch
import random
from torch import Tensor
from Shakespeak.utils import CharDataSet


class N_gram:
    '''
    Simple 2 or 3 gram model baseline.
    '''
    def __init__(self, N: int, dataset_path: str, N_tokens: int = 128):
        '''
        N is the order of the N_gram : 2 or 3. N_tokens influences the way 
        the fold are separated. Initialize N_tokens as in the transformer 
        model to get the same exact folds, N_tokens is just used in this 
        classes's CharDaset for validation and train data. This data
        directly reads out the validation / train indices from the data
        loaders, it does not subscript them.
        '''
        self.dataset_path = dataset_path
        self.N_tokens = N_tokens
        self.N = N                      # order of the N_gram model
        self.epsilon = 1e-10            # probability for unseen n-grams
        self.decoder = {}               # decoder dict to be able to convert indices back to their corresponding char
        self.train: CharDataSet = None
        self.valid: CharDataSet = None
        self.ngrams: dict[int, dict[int, int]] | dict[int, dict[int, dict[int, int]]] = {}      
        '''dict of word frequencies, in bigram ngram[wi][wi+1] = #times (wi wi+1) appear in train set'''
    
    def train_model(self, fold: int, k_fold: int) -> tuple[float, float]:
        '''
        Returns
        -------
        (perplexity_train, perplexity_validation)
        '''
        self.ngrams = {}
        self.train = CharDataSet(self.N_tokens, fold, k_fold, self.dataset_path, is_training=True)
        self.decoder = self.train.decoder 
        train_idx = self.train.train_idx         # get the tokens
        
        self.valid = CharDataSet(self.N_tokens, fold, k_fold, self.dataset_path, is_training=False)
        val_idx = self.valid.valid_idx             # get the tokens

        if self.N == 2 :                         # Bigram
            for i in range(len(train_idx)-1):    # for every train indice
                w1 = train_idx[i].item()
                w2 = train_idx[i+1].item()
                if not w1 in self.ngrams:
                    self.ngrams[w1] = {}
                if not w2 in self.ngrams[w1]:
                    self.ngrams[w1][w2] = 1
                else:
                    self.ngrams[w1][w2] += 1     # increment count
        
            for c in self.ngrams: # go through keys
                self.ngrams[c] = dict(sorted(self.ngrams[c].items(), key=lambda item: -item[1])) # sort from most frequent
                total = sum(self.ngrams[c].values())                                             # sum all the count, total = count(c) in train text
                self.ngrams[c] = dict([(k, self.ngrams[c][k]/total) for k in self.ngrams[c]])    # compute relative frequency P(A | B) = P(A, B)/P(B)
        
        if self.N == 3:
            for i in range(len(train_idx)-2):
                w1 = train_idx[i].item()
                w2 = train_idx[i+1].item()
                w3 = train_idx[i+2].item()

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
        perplexity_train = self.evaluate(train_idx) # eval on training dataset
        perplexity_validation = self.evaluate(val_idx) #eval on validation dataset
        return perplexity_train, perplexity_validation
               
    def evaluate(self, data_idx: Tensor) -> float: # training perplexity
        perplexity = 1 # neutral element for multiplication
        if self.N == 2:
            total_pairs = len(data_idx) - 1
            for i in range(total_pairs):
                current_word = data_idx[i].item()
                next_word = data_idx[i + 1].item()
                # Calculer la probabilité prédite P(w_{i+1} | w_i)
                try:
                    P_wi_plus_1_wi = self.ngrams[current_word][next_word]
                except: # caractère pas présent dans les données de training
                    P_wi_plus_1_wi = self.epsilon   
                # Calculer la perplexité pour cette paire de mots
                perplexity *= math.pow(1/P_wi_plus_1_wi, 1/total_pairs)

        if self.N == 3:
            total_triples = len(data_idx) - 2
            for i in range(total_triples):
                current_word = data_idx[i].item()
                next_word = data_idx[i + 1].item()
                next_next_word = data_idx[i + 2].item()
                # Calculer la probabilité prédite P(w_{i+2} | w_{i+1}, w_i)
                try:
                    P_wi_plus_2_wi_plus_1_wi = self.ngrams[current_word][next_word][next_next_word]
                except: # suite de caractères pas présent dans les données de training
                    P_wi_plus_2_wi_plus_1_wi = self.epsilon
                
                # Calculer la log loss pour cette paire de mots
                perplexity *= math.pow(1/P_wi_plus_2_wi_plus_1_wi,1/total_triples)
        return perplexity
        
    def next_number(self, n1, n2 = 0): # returns next index generated
        '''
        Sample next vocabulary index based on the ngram distribution of (n1, n2) (bigram) or (n1, n2, n3) (trigram).
        '''
        if self.N == 2:
            vars = self.ngrams[n1]
            return random.choices(list(vars.keys()), weights=vars.values())[0]
        if self.N == 3:
            vars = self.ngrams[n1][n2]
            return random.choices(list(vars.keys()), weights=vars.values())[0]

    def generate(self, n_new_tokens, char_idx1, char_idx2 = 0) -> Tensor:
        '''
        Function that generates a list of int representing the characters.
        '''
        if self.N == 2:
            gen_idx = [char_idx1]
            n1 = char_idx1
    
            for _ in range(n_new_tokens):
                n2 = self.next_number(n1) # generate knowing n1
                gen_idx.append(n2)
                n1 = n2
            return Tensor(gen_idx)
    
        if self.N == 3:
            gen_idx = [char_idx1, char_idx2]
            n1 = char_idx1
            n2 = char_idx2
    
            for _ in range(n_new_tokens):
                n3 = self.next_number(n1,n2) # generate knowing n1 and n2
                gen_idx.append(n3)
                n1 = n2
                n2 = n3
            return Tensor(gen_idx)


def cross_val_ngram(N: int, k_fold: int, path: str, N_tokens: int = 128) -> tuple[float, float]:
    """
    Run a k_fold cross validation experiment with a N-gram model,
    return the validation perplexity mean and standard deviation. 
    """
    baseline = N_gram(N, path, N_tokens)

    perx_val = torch.zeros(size=(k_fold,))
    perx_tra = torch.zeros(size=(k_fold,))

    for k in range(k_fold):
        perx_train, perx_valid = baseline.train_model(k, k_fold) # ~5 seconds per call
        perx_val[k] = perx_valid
        perx_tra[k] = perx_train

    perx_val_std, perx_val_mean = perx_val.std(-1).item(), perx_val.mean(-1).item()
    return perx_val_mean, perx_val_std

if __name__ == "__main__":
    dataset = "./datasets/shakespear_corpus.txt"
    N_tokens = 64                                   # required to fold the dataset...
    bigram_model = N_gram(2, dataset, N_tokens)
    trigram_model = N_gram(3, dataset, N_tokens)
    
    # Single training 
    b_perplexity_train, b_perplexity_validation = bigram_model.train_model(fold=1, k_fold=10)
    t_perplexity_train, t_perplexity_validation = trigram_model.train_model(fold=1, k_fold=10)
    
    # Generate numbers
    start_char_1 = 44 # start for bigram
    start_char_2 = 52 # start for trigram

    generation_size = 300
    
    bigram_text = bigram_model.train.decode(bigram_model.generate(generation_size, start_char_1))
    trigram_text = trigram_model.train.decode(trigram_model.generate(generation_size, start_char_1,start_char_2))
    
    print("Bigram text generation \n---------------------------")
    print(bigram_text + '\n')
    print("Trigram text generation \n--------------------------")
    print(trigram_text)
    