import json

with open('qnaFinal.json', 'r', encoding='utf-8') as json_data:
    question_intents = json.load(json_data)

dict={}
tag=''

for data in question_intents:
    print(data)


