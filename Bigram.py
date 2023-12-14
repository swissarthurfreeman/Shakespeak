import re
import random

def next_char(char):
    vars = bgrams[char]  
    return random.choices(list(vars.keys()), weights=vars.values())[0]

def generate_text():
    chars = []
    c = '.'
    for i in range(50):
        c = next_char(c)
        if c == '.':
            break
        chars.append(c)
    return (' '.join(chars) + '. ').capitalize()

def load_data(filename) -> str:
    with open(filename, 'r') as file:
        data = file.read() # read data
        data = data.lower().replace("\n", ' ') # lower case
        data = re.sub('[^a-z0-9\.\-]', ' ', data) # remove non-alpha_numeric
        data = re.sub('\s{2,}', ' ', data) # remove multiple spaces
        data = '. ' + data.replace('.', ' .') # 
        data = re.sub(' [a-z0-9] ', '', data) # remove single char (reduce noise)
    return data


if __name__ == '__main__':
    RAW_DATA_PATH = './datasets/shakespear_corpus.txt'

    preprocessed_text = load_data(RAW_DATA_PATH)
    
    characters = list(preprocessed_text) # split text to char list
    bgrams = {}
    for i in range(len(characters)-2):
        w1 = characters[i]
        w2 = characters[i+1]
        if not w1 in bgrams:
            bgrams[w1] = {}
        if not w2 in bgrams[w1]:
            bgrams[w1][w2] = 1
        else:
            bgrams[w1][w2] += 1
        
    for c in bgrams: # go through keys
        bgrams[c] = dict(sorted(bgrams[c].items(), key=lambda item: -item[1])) # sort from most frequent
        total = sum(bgrams[c].values()) # sum all the count
        bgrams[c] = dict([(k, bgrams[c][k]/total) for k in bgrams[c]]) # compute relative frequency

    
    #* Generate characters
    
    generated_text = []
    for i in range(5):
        generated_text.append(generate_text())
    
    
    print(generated_text)
    
