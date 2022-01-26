from difflib import get_close_matches
import json
from random import choice
import re
from speech import speak
from web import stopword

data = json.load(open('dict_data.json', encoding='utf-8'))

def dictionary(query):
    list = re.split("\W+", query)
    text = [word for word in list if word not in stopword]
    keyword = ' '.join(map(str, text))
    word, result, check = getMeaning(keyword)
    print(result)
    result = choice(result)
    if check == 1:
        speak(f'The definition of {word} is {result}')
        # return ["Here's the definition of \"" + word.capitalize() + '"', result]
    elif check == 0:
        speak(f"I think you are looking for {word}. It's definition is {result}")
        print(result)
        # return ["I think you're looking for \"" + word.capitalize() + '"', "It's definition is,\n" + result]
    else:
        speak(f"{result}")
        # return [result, '']


def getMeaning(word):
    if word in data:
        return word, data[word], 1
    elif len(get_close_matches(word, data.keys())) > 0:
        word = get_close_matches(word, data.keys())[0]
        return word, data[word], 0
    else:
        return word, ["This word doesn't exists in the dictionary."], -1


# print(len(data))