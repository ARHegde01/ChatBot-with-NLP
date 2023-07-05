# Developer : Anudeep Hegde
# Date : 07/04/23 - 


# All necessary imports w/o GUI 
import random
import json 
import pickle
import numpy as np 
import nltk 

from nltk.stem import WordNetLemmatizer
from keras.models import load_model


# Initialize the lemmatizer object 
Lemmatizer = WordNetLemmatizer()

# Reads the intents.json file and loads it into a python dictionary 
Load_Intents = json.loads(open('intents.json').read())
Words = pickle.load(open('words.pkl', 'rb'))
Classes = pickle.load(open('classes.pkl', 'rb'))

# Load the pre trained model

Model = load_model("chatbot_model.h5")


## Defining Functions


# he clean_up_sentence function tokenizes a given sentence, performs lemmatization on the words, and returns a list of the resulting lemmas.

def clean_up_sentence(sentence):
    words_in_sentence = nltk.word_tokenize(sentence)
    words_in_sentence = [Lemmatizer.lemmatize(word) for word in words_in_sentence]

    return words_in_sentence

# The bag_of_words function generates a binary vector representation of a given sentence based on a predefined set of words.

def bag_of_words(sentence):
    words_in_sentence = clean_up_sentence(sentence)
    bag = [0] * len(Words)

    for w in words_in_sentence:
        for i, word in enumerate(Words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# The predict_class function predicts the intent of a given sentence using a model, returning a list of intents and their probabilities based on the prediction.

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = Model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25

    Results = [[i , r] for i , r in enumerate(res) if r > ERROR_THRESHOLD]
    Results.sort( key = lambda x:x[1], reverse = True)

    Return_List = []

    for r in Results:
        Return_List.append({'intent': Classes[r[0]], 'probability' : str(r[1])})

    return Return_List

# This function retrieves a randomly selected response from a list of intents in a JSON object by matching the tag of the first intent in a provided list.
def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result 
print("Bot is now active")

while True:
    message = input("")
    ints = predict_class(message)
    if ints:
        res = get_response(ints, Load_Intents)
        print(res)




