import numpy as np
import random
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('dict_data.json', 'r', encoding="utf8") as f:
    dict_intents = json.load(f)

print(dict_intents)


