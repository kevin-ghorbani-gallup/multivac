import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from nltk.corpus import stopwords
import string
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate

import tensorflow as tf


def load_data(filename):
    '''loads text data'''
    with open(filename, 'r') as f:
        lines = f.read()
    return lines

def clean_doc(doc):
    '''data cleaning'''
    # tokenize
    tokens_ = doc.split()
    
    # remove punctuations
    table_ = str.maketrans('', '', string.punctuation)
    tokens_ = [word_.translate(table_) for word_ in tokens_]
    
    # remove non-alphabetics
    tokens_ = [word_ for word_ in tokens_ if word_.isalpha()]
    
    # remove stopwords
    stop_words_ = set(stopwords.words('english'))
    tokens_ = [word_ for word_ in tokens_ if not word_ in stop_words_]
    
    # word limit (can be improved)
    tokens_ = [word_ for word_ in tokens_ if len(word_) > 1]
    
    return tokens_


def tokenize_sentence(sentence):
    tokens_ = word_tokenize(sentence)
    words_ = [word_ for word_ in tokens_ if word_.isalpha()]
    return words_

def tokenize_words_in_sentences(sentences):
    sentences_token = []
    for sentence in sentences:
        sentences_token.append(tokenize_sentence(sentence))
        
    return sentences_token

def save_dataset(dataset, file_name):
    '''dumps data in a pickle file'''
    pickle.dump(dataset, open(file_name, 'wb'))
    print('Saved: %s' % file_name)
    
def load_dataset(file_name):
    '''reads data from pickle file'''
    return pickle.load(open(file_name, 'rb'))

def create_tokenizer(lines):
    '''fit a tokenizer'''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
 
def max_length(lines):
    '''calculates the maximum document length'''
    return max([len(s) for s in lines])
 
def encode_text(tokenizer, lines, length):
    '''encodes a list of lines'''
	# integer encode
    encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


def corrupt_sentence_generator(input_tokenized_lines, size=None):
    '''generates sequence of random words
    size: default is None, generates a random-sized sequence'''
    sentence_ = []
    if size is None:
        size = len(np.random.choice(input_tokenized_lines))
    for _ in range(size):
        sentence_.append(np.random.choice(np.random.choice(input_tokenized_lines)))
    return sentence_


def define_model(length, vocab_size):
    '''Model is created'''
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # interpretation
    dense1 = Dense(10, activation='relu')(flat1)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=inputs1, outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model

