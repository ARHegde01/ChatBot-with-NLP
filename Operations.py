# Developer : Anudeep Hegde
# Date : 07/04/23 - 


# All necessary imports w/o GUI 

import random
import json
import pickle 
import pickle 
import numpy as np 
import tensorflow as tf 

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

## Code Begins Here: ##


# Initialize Variables

Lemmatizer = WordNetLemmatizer()
Load_Intents = json.loads(open('intents.json').read())

Words = []
Classes = []
Docs = []
IgnoredLetters = ['?', '!', '.', ',']


# Tokenize patterns into a list of words
# Using nltk.word_underscore()

for intent in Load_Intents['intents']:
    for pattern in intent['patterns']:

        WordList = nltk.word_tokenize(pattern)
        Words.extend(WordList)
        Docs.append((WordList, intent['tag']))

        if intent['tag'] not in Classes:
            Classes.append(intent['tag'])

# Lemmatize each word in Words
# Using Lemmatizer 

Words = [ Lemmatizer.lemmatize(word) for word in Words if word not in IgnoredLetters]
Words = sorted(set(Classes))

# Sort in alphabetical order and remove any duplicates
Classes = sorted(set(Classes))

pickle.dump(Words, open('words.pkl', 'wb', ))
pickle.dump(Classes, open( 'classes.pkl', 'wb'))


## Create Training Data

Training_Data = []
outputEmpty = [0] * len(Classes)

for document in Docs:
    bag = []
    WP = document[0]
    WP = [Lemmatizer.lemmatize(word.lower()) for word in WP ]

    for word in Words:
        if word in WP:
            bag.append(1)
        else:
            bag.append(0)
    
    outputRow = list(outputEmpty)
    outputRow[Classes.index(document[1])] = 1
    Training_Data.append(bag + outputRow)

random.shuffle(Training_Data)
Training_Data = np.array(Training_Data)

TrainX = Training_Data[: , : len(Words)]
TrainY = Training_Data[: , len(Words): ]

# Designing the Neural Network Model
# This NNM has 128 units and the activation function used in this layer is 'relu'

Model = tf.keras.Sequential()

Model.add(tf.keras.layers.Dense(128, input_shape = (len( TrainX[0] ) ,  ), activation = 'relu') )

# Adding a dropout layer to the model
# This will randomly set 50% of the inputs to 0 at each update during training 
# To reduce overfitting 

Model.add(tf.keras.layers.Dropout(0.5))
Model.add(tf.keras.layers.Dense(64, activation = 'relu'))
Model.add(tf.keras.layers.Dense(len(TrainY[0]), activation = "softmax"))

# At this point, the above adds the final densely connecting neural network layer to the model

sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)

# Compile the model

Model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Save the training history
history = Model.fit(np.array(TrainX), np.array(TrainY), epochs = 200, batch_size = 5, verbose = 1)

# Save the training order to a file
Model.save('chatbot_model.h5',  history)

print("Executed: Model has been trained and saved successfuly")








