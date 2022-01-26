import json

# with open('qnaFinal.json', 'r', encoding='utf-8') as json_data:
#     test_intents = json.load(json_data)

with open('qna.json', 'r', encoding='utf-8') as json_data:
    qna_intents2 = json.load(json_data)

singleQnAs = []
id = 1
dict = {}

for data in qna_intents2:
    question=data['question']
    answer=data['annotations'][0]['answer']
    dict[question]=answer

# for data in test_intents:
#     question = data['question']
#     answers = data['annotations'][0]['answer']
#     dict[question]=answers

print(dict)
# print(len(dict))

