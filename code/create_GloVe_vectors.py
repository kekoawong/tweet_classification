import zipfile
import numpy as np
import pickle

# declare dictionary object
word_vectors = {}
file_name = '../data/gloVe_vectors.obj'

# parse zipped file line by line
with zipfile.ZipFile('../../glove.twitter.27B.zip') as z:
    with z.open('glove.twitter.27B.100d.txt') as f:
        
        # iterate through each line
        for index, line in enumerate(f):
            values = line.split()
            word = values[0].decode('utf-8')
            vector = np.asarray(values[1:], "float32")
            word_vectors[word] = vector
            if index % 100000 == 0:
                print(f'On word {word} on line {index}')

print('Writing to file')
with open(file_name, 'wb') as out_file:
    pickle.dump(word_vectors, out_file)
print('Done')