import random

import torch
import json
from brain import replyFunction
from model import NeuralNet
from nltk_utils import *
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
# print(all_words)
# print(len(all_words))
tags = data['tags']
# print(tags)
# print(len(tags))

model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

test_data = ['What is the temperature of goa today',
             "Can you tell me today's date",
             'How is nifty',
             'At what time do you open?',
             'I need help',
             'What day is it?',
             'How do you do']

with open('test.json', 'r') as json_data:
    test_intents = json.load(json_data)

# print(test_intents)

xy = []

# loop through each sentence in our intents patterns
for intent in test_intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        # w = tokenize(pattern)
        # add to our words list
        # all_words.extend(w)
        # add to xy pair
        xy.append((pattern, tag))

print(xy)

for pattern_sentence, tag in xy:
    # print("Sentence=" + pattern_sentence)
    # print("Corresponding Tag=" + tag)

    sentence = tokenize(pattern_sentence)

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)

    predicted_tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    print("Original Sentence = " + pattern_sentence)
    print("Original Tag = " + tag)

    print(f"Prediction Confidence = {prob.item()}")
    print("Predicted Tag = " + tag)

    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(random.choice(intent['responses']))
                # return tag
    else:
        print('I could not understand you')

    print('\n')


def testFuntion():
    for sentence in test_data:
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)

        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        print("Original Sentence = " + sentence)
        print("Original Tag = " + tag)

        print(f"Prediction Confidence = {prob.item()}")
        print("Predicted Tag = " + tag)

        # print(prob.item())
        # print(tag)

        # if prob.item() > 0.75:
        #     for intent in intents['intents']:
        #         if tag == intent["tag"]:
        #             # return random.choice(intent['responses'])
        #             return tag
        # else:
        #     return 'I could not understand you'

# sys.exit(0)

# testFuntion()
