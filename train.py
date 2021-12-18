import numpy as np
import random
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

writer = SummaryWriter("runs/voiceassistant")

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Training Pipeline
# Feed Forward Nueral Network - Linear Layer
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] #Number of patterns

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!', "'s", ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
# print(X_train[0])
# print(y_train[0])

# Hyper-parameters
num_epochs = 400
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):

    def __init__(self):
        # data loading
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
# print(dataset)
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
# print(train_loader)

# sys.exit(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# writer.add_graph(model, X_train.reshape(-1,139))
# writer.close()
# sys.exit()

# Train the model
n_total_steps = len(train_loader)
# print(n_total_steps) #13
running_loss = 0.0
running_correct = 0

Loss = []
Epoc = []
accuracy = []

for epoch in range(num_epochs):
    for i, (words, labels) in enumerate(train_loader):
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # Loss calculation
        loss = criterion(outputs, labels)
        # Clear the gradients
        optimizer.zero_grad()
        # Credit assignment
        loss.backward()
        # Update Model Weights
        optimizer.step()

        #To plot the graph
        Loss.append(loss.item()*10)
        Epoc.append(epoch)
        accuracy.append(100 - loss.item()*10)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)

        running_correct += (predicted == labels).sum().item()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        # print('Training Loss', running_loss / 100, epoch * n_total_steps + i)
        # print('Accuracy', running_correct / 100, epoch * n_total_steps + i)
        running_loss = 0.0
        running_correct = 0

# print(Loss)
# print(accuracy)
print(f'final loss: {loss.item():.4f}')

plot1 = plt.figure(1)
plt.title("Loss vs Epoch")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(Epoc, Loss)

plot2 = plt.figure(2)
plt.title("Accuracy vs Epoch")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(Epoc, accuracy)

plt.show()

# Storing the data from the model as a dictionary
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
# torch.save uses Python's pickel module to serialize the objects and saves them
# The result is serialized and not human readable
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
