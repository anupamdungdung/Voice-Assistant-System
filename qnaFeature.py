from difflib import get_close_matches
import json
from random import choice
import re
from speech import speak
from web import stopword

data = json.load(open('qnaFinal.json', encoding='utf-8'))


def qna(query):
    list = re.split("\W+", query)
    text = [word for word in list]
    question = ' '.join(map(str, text))
    word, answer, check = getAnswer(question)
    print(answer)
    answer = choice(answer)
    if check == 1:
        speak(f'{answer}')
        # return ["Here's the definition of \"" + word.capitalize() + '"', result]
    elif check == 0:
        speak(f"It's answer is {answer}")
        # return ["I think you're looking for \"" + word.capitalize() + '"', "It's definition is,\n" + result]
    else:
        speak(f"{answer}")
        # return [result, '']


def getAnswer(question):
    if question in data:
        return question, data[question], 1
    elif len(get_close_matches(question, data.keys())) > 0:
        word = get_close_matches(question, data.keys())[0]
        return word, data[word], 0
    else:
        return question, ["This question doesn't exist in the database."], -1

print(len(data)) #2899 questions