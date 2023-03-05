'''
Programming Assignment 4
Pranaav Rao
This python program implements a Decision List classifer to perform word
sense disambiguation. The main feature in use is the Bag-of-Words model.
my-model.txt will display each feature, the log-likelihood associated with
it, and the sense it predicts.
'''
import math
from sys import argv
import re
from nltk.corpus import stopwords
import json

#used later for preprocessing
def num_there(s):
    return any(i.isdigit() for i in s)

train = ''
test = ''

#for preprocessing
punc = '''.,-?!:;_[]{}()'"|\&$%'''
word = ['<head>line</head>','<head>lines</head>']
special_characters = ['<@>','<p>','</s>','<s>','</p>']

#open line-train into string
with open(argv[1], 'r') as f:
    train += f.read()

with open(argv[2], 'r') as f:
    test += f.read()

#split by instance
train_instances = train.replace('\n','').split('</instance>')

#map each context word to the associated sense while recording frequency
phone_sense = {}
product_sense = {}
for instance in train_instances:
    if "id=" in instance:
        c_match = re.search('<context>(.*)</context>', instance)
        context = c_match.group(1)
        #pre-processing
        # 1. remove punctuation
        # 2. remove special characters
        # 3. remove numerics
        # 4. remove stop words
        context = context.lower()
        for ptr in context:
            if ptr in punc:
                context = context.replace(ptr, "")
        sentenceWords = context.split()
        for i in sentenceWords[:]:
                if i in word or i in special_characters or num_there(i) or i in set(stopwords.words('english')):
                    sentenceWords.remove(i)
        if 'senseid="phone"' in instance:
            for i in sentenceWords:
                if i in phone_sense:
                    phone_sense[i] += 1
                else:
                    phone_sense[i] = 1
        elif 'senseid="product"' in instance:
            for i in sentenceWords:
                if i in product_sense:
                    product_sense[i] += 1
                else:
                    product_sense[i] = 1

# dictionary that records frequency of phone and product for each word
# bag of words model
word_sense = {}
for i in phone_sense:
    word_sense[i] = {'phone':phone_sense[i]}
for i in product_sense:
    word_sense[i] = {'product':product_sense[i]}
for i in word_sense:
    # 0.001 to work with the log-likelihood formula
    if 'product' not in word_sense[i].keys():
        word_sense[i]['product'] = 0.001
    elif 'phone' not in word_sense[i].keys():
        word_sense[i]['phone'] = 0.001
    #log-likelihood
    word_sense[i]['likelihood'] = abs(math.log(word_sense[i]['phone']/word_sense[i]['product']))
    #sense
    if word_sense[i]['phone'] > word_sense[i]['product']:
        word_sense[i]['sense'] = 'phone'
    else:
        word_sense[i]['sense'] = 'product'

with open(argv[3], 'w') as f:
    f.write('DECISION LIST MODEL:\n')
    f.write(json.dumps(word_sense))       


#retrieve instances from test
test_instances = test.replace('\n','').split('</instance>')
#retrieve sentences for each id
test_sentences = []
for instance in test_instances:
    if "id=" in instance:
        c_match = re.search('<context>(.*)</context>', instance)
        sentence = c_match.group(1)
        #pre process
        sentence.lower()
        for ptr in sentence:
            if ptr in punc:
                sentence = sentence.replace(ptr, "")
        sentenceWords = sentence.split()
        for i in sentenceWords[:]:
                if i in word or i in special_characters or num_there(i) or i in set(stopwords.words('english')):
                    sentenceWords.remove(i)
        sentence = ' '.join(sentenceWords)
        test_sentences.append(sentence)

#make predictions on test sentence
for i in test_sentences:
    count = 0
    i_words = i.split()
    for word in i_words:
        if word in word_sense:
            if word_sense[word]['sense'] == 'product':
                count += 1
            elif word_sense[word]['sense'] == 'phone':
                count -= 1
    if count > 0:
        print('product')
    elif count <= 0:
        print('phone')
