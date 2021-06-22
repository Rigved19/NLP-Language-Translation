import re
import copy
import pickle
from matplotlib import pyplot as plt
from pickle import dump
from unicodedata import normalize
from numpy import array
import string 
import numpy as np
from numpy import asarray
from keras.preprocessing.sequence import pad_sequences
from numpy import zeros
 
#Cleaning and Processing Data (English and German sentences)
input_sentences = []
output_sentences = []
output_sentences_inputs = []
clean_input_sentences = []
clean_output_sentences = []
clean_output_sentences_inputs = []
 
NUM_SENTENCES =1000   #Taking the first 1000 sentences as our examples
count = 0
 
#Adding <eos> and <sos> for corresponding output sequences
#<sos> is "start of sentence" and <eos> is "end of sentence" for Output Sequences
for line in open(r'/content/deu.txt', encoding="utf-8"):
    count += 1
 
    if count > NUM_SENTENCES:
        break
 
    input_sentence, output , extra = line.rstrip().split('\t')
    
    output_sentence = output + ' <eos>'
    output_sentence_input = '<sos> ' + output
 
    input_sentences.append(input_sentence)
    output_sentences.append(output_sentence)
    output_sentences_inputs.append(output_sentence_input)
 

#Cleaning all the data,seperating and storing
def clean_pairs(lines):
    cleaned = list()
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        clean_pair = list()
        # tokenize on white space
        line = line.split()
        # convert to lowercase
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [word.translate(table) for word in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)
 
clean_input_sentences =  clean_pairs(input_sentences)
clean_output_sentences = clean_pairs(output_sentences)
clean_output_sentences_inputs = clean_pairs(output_sentences_inputs)
 
print("Printing processed English Sentences : "  , clean_input_sentences[959])
print("Printing corresponding processed Gemran Input Sentences : " ,  clean_output_sentences_inputs[959])
print("Printing Gemran Output Sentences" , clean_output_sentences[959])

#Counting no. of sentences(examples) in 
input_examples = len(clean_input_sentences)
output_examples = len(clean_output_sentences)

print("\nNumber if English Sentences" , input_examples)
print("Number if German Sentences" , output_examples)


#Creating dictionary for English Vob and German Vocab
#Contains a dictionary with unique words as keys with index numbers as values

def make_dict(sentences):
  set_input = set()              #Set() helps to count multiple same words only once
  dict_vocab = {}
  dict_index = {}

  for i in sentences:
    for j in i:
      words = j.split()
      set_input.update(words) 
  
  list_input = sorted(list(set_input))  # .sorted() - Sorting the list of Words in Ascending order
  vocab_size = len(list_input)
  

  for i in range(vocab_size):         #Forming Vocab Dictictionaries ( Words --> index)
    dict_vocab[list_input[i]] = i+1   # i+1 so that it can be read easier , because of 0 index

  for i in range(vocab_size):         #Forming Vocab Dictictionaries ( index --> Words)
    dict_index[i+1] = list_input[i]
 

  return(vocab_size,dict_vocab,dict_index)

#Forming English/German ( Vocab, Index, Size)
vocab_eng_size , vocab_eng , index_eng = make_dict(clean_input_sentences)
vocab_german_size, vocab_german , index_german = make_dict(clean_output_sentences)

#Adding the extra word "sos" in the german dictionary
vocab_german_size = vocab_german_size + 1
vocab_german["sos"] = vocab_german_size
index_german[vocab_german_size] = "sos"


print("\nEnlish Word-->Index :",vocab_eng)
print("\nEnlish Index-->Word :",index_eng)
print("\nGerman Word-->Index :",vocab_german)
print("\nGerman Index-->Word :",index_german)
print("\nEng Dictionary Size: ",vocab_eng_size)
print("German Dictionary Size:",vocab_german_size)
idx_sos = vocab_german["sos"]                      #Storing <sos> token number, used for sampling test cases
idx_eos = vocab_german["eos"]                      #Storing <eos> token number, used for sampling test cases


#Finding maximum length of sentence in English and German sentences
def max_length(sentences):
  max_length = 1
  for i in sentences:
    for j in i:
      words = j.split()
      length = len(words)

    if length >= max_length:
      max_length = length
    else:
      pass

  return(max_length)

max_eng_length = max_length(clean_input_sentences)
max_german_length = max_length(clean_output_sentences)

print("Length of longest English sentence :", max_eng_length)
print("Length of longest German sentence :", max_german_length)


#Tokenization and Padding

def tokenization(sentences,para):
  final_token = []
  for i in sentences:
    for j in i:
      words = j.split()
      token = []
      for x in words:
        if para ==0:
          index = vocab_eng[x]
        else:
          index = vocab_german[x]
        token.append(index)
    final_token.append(token)
  
  return(final_token)

token_eng = tokenization(clean_input_sentences,0)
token_german_inp = tokenization(clean_output_sentences_inputs,1)
token_german_out = tokenization(clean_output_sentences,1)

print("After tokenization English Sentences = ", token_eng[959])
print("After tokenization German Input Sentences = ", token_german_inp[959])
print("After tokenization German Output = ", token_german_out[959])


#Pre padding for Input sentences and Post padding for Output sentences
def padding(token,para):
  thislist = []
  if para == 0:
    length = max_eng_length
    for x in token:
      pad_arr = np.pad(x , (length-len(x),0) , 'constant', constant_values=0)
      arrlist = pad_arr.tolist()
      thislist.append(arrlist)
      
  else:
    length = max_german_length
    for x in token:
      pad_arr = np.pad(x , (0,length-len(x)) , 'constant', constant_values=0)   
      arrlist = pad_arr.tolist()
      thislist.append(arrlist)
      
  return(thislist)


padded_eng = padding(token_eng,0)
padded_german_inp = padding(token_german_inp,1)
padded_german_out = padding(token_german_out,1)

print("\nAfter Padding English Sentences = " , padded_eng[959])
print("After Padding Gemrman Input Sentences = " , padded_german_inp[959])
print("After Padding Gemrman Output Sentences = " , padded_german_out[959])
    
#Word Embedding for Eng lang input layer

embeddings_dictionary = {}

glove_file = open(r'/content/glove.6B.50d.txt', encoding="utf8")

#Reading the required words from the emedding file and storing the required in code for further initialisation
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

"""Creating a embedding matrix with row number is the integer value of the word and the column will represent corresponding 
embedding vector for the word
"""
embedding_matrix = np.zeros((vocab_eng_size+1, 50))   #Index 0 will have no values 
for word, index in vocab_eng.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

        
"""
Creating Decoder Input and Output layer (one hot) for german language with row number is the integer value of the word and the 
column will represent corresponding one hot vector for the word with 1 value at the index position 
corresponing to the integer value of the word
"""
german_matrix = np.zeros((vocab_german_size+1, vocab_german_size))
for word, index in vocab_german.items():
     german_matrix[index,index-1] = 1
     


# Labeling each important variable for reference

input_examples #No. of English Senteces
output_examples #No.of German Sentences = above
vocab_german_size #No. of German words
max_eng_length #Encoder Time Steps
max_german_length #Decoder Time Steps
embedding_matrix #Input Embedding Matrix
german_matrix #Output German Matrix

padded_eng = np.array(padded_eng) #X_enc_train
padded_german_inp = np.array(padded_german_inp) #X_dec_train
padded_german_out = np.array(padded_german_out) #Ytrain

#Forming Input 3D Tensors

#Converting the input sequence into embedding and reshaping a 3D tensor of shape(Embedding dim, Number of sentences, Timestep)
Xenc_train=np.zeros((50, input_examples , max_eng_length))
for m in range(input_examples):
    for t in range(max_eng_length):
        Xenc_train[:,m,t]=embedding_matrix[padded_eng[m,t]]

#Converting the output sequence into one-hot and reshaping a 3D tensor of shape(ger_vocab_size, Number of sentences, Timestep)
Xdec_train = np.zeros((vocab_german_size, input_examples , max_german_length))
for m in range(input_examples): 
  for t in range(max_german_length):
    Xdec_train[:,m,t] = german_matrix[padded_german_inp[m,t]]

Ydec_train = np.zeros((vocab_german_size, input_examples , max_german_length))
for m in range(input_examples): 
  for t in range(max_german_length):
    Ydec_train[:,m,t] = german_matrix[padded_german_out[m,t]]

print(Xenc_train[:,959,:])