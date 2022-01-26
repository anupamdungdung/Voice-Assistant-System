import speech_recognition as sr
from googletrans import Translator
import sys
import random
from threading import Thread
from tkinter import *

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
AITaskStatusLblBG = '#203647'

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

from web import *
from os_operations import *
from tasks import *
from dictionary import *
from math_operations import perform
from model import *
from nltk_utils import *
from fileHandler import *
from qnaFeature import *

# get audio from the microphone
r = sr.Recognizer()
r.energy_threshold = 1500
translator = Translator()

test_data = ['How is nifty', 'How much do you charge?', 'What are your policies?', 'At what time do you open?',
             'Help me', 'What day is it?', 'I feel lonely']


def hour():
    h = datetime.datetime.now().hour
    if 1 <= h < 12:
        speak("Good Morning!")
    elif 12 <= h < 18:
        speak("Good Afternoon!")
    else:
        speak("Good evening!")


def greeting():
    reply = translator.translate("Hello, how can I help you?", dest='en')
    speak(reply.pronunciation)


def exitTheApp():
    reply = translator.translate("Bye, See you soon", dest='en')
    # print(reply)
    speak(reply.pronunciation)
    sys.exit(0)


def isContain(txt, lst):
    for word in lst:
        if word in txt:
            return True
    return False


def replyFunction(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    # print(X)
    output = model(X)
    # print(output)
    # value, index
    _, predicted = torch.max(output, dim=1)
    # print("Tags Shape="+tags.shape[0])
    # print(predicted)
    tag = tags[predicted.item()]
    print(tag)
    # Here we have to implement the softmax function manually
    probs = torch.softmax(output, dim=1)
    # print(probs)
    # print(predicted.item())
    prob = probs[0][predicted.item()]
    print(prob.item())

    if prob.item() > 0.65:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # return random.choice(intent['responses'])
                return tag
    else:
        speak('I could not understand you')


####### SET UP TEXT TO SPEECH #######

def speak(audio):
    try:
        engine.say(audio)
        # print(f"Bot said: {audio}")
        engine.runAndWait()
    except Exception as e:
        print(e)


####### SET UP SPEECH TO TEXT #######

def record():
    print('Listening...')
    r = sr.Recognizer()
    r.energy_threshold = 1500
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""
        try:
            said = r.recognize_google(audio)
            said = translator.translate(said)
            # print(said)
            f = open('learn.txt', 'a')
            f.write(said.text+'\n')
            f.close()
            print(f"User said: {said.text}")
        except Exception as e:
            print(e)
            if "connection failed" in str(e):
                speak("Your System is Offline")
    return said.text.lower()


def voiceMedium():
    while True:
        try:
            # Thread(target=voiceMedium()).start()
            # query = Thread(target=record()).start()
            query = record()
            if query is None:
                pass
            else:
                main(query.lower())
            # if 'Priya' in callout:
            #     query=record()
            #     if query is None:
            #         pass
            #     else:
            #         main(query.lower())
        except Exception as e:
            # speak('Sorry! I could not understand you!')
            pass


def main(query):
    if 'flipkart' in query:
        speak('Opening Flipkart')
        flipkart(query)
    elif 'amazon' in query:
        speak('Opening Amazon')
        amazon(query)
    elif 'youtube' in query:
        speak('Opening Youtube')
        youtube(query)
    elif 'open calculator' in query:
        speak('Opening Calculator')
        openCalculator(query)
    elif 'open notepad' in query:
        speak('Opening Notepad')
        openNotePad(query)
    elif 'close notepad' in query:
        speak('Closing Notepad')
        closeNotePad()
    elif 'open camera' in query:
        speak('Accessing your device camera')
        openCamera()
    elif 'close camera' in query:
        speak('Closing device camera')
        closeCamera()
    elif 'joke' in query:
        speak("Here's a joke for you")
        jokes()
    elif 'whatsapp' in query:
        speak("Sure")
        speak("Whom do you want to send the message?")
        phoneNo = record()
        speak('What is the message?')
        message = record()
        sendWhatsapp(phoneNo, message)

    elif 'create' in query and 'file' in query or 'document' in query:
        createFile(query)

    elif isContain(query, ['web']):
        speak('Let me search the web for you')
        openWebsite(query)

    elif isContain(query, ['map', 'direction', 'directions']):
        if "direction" in query:
            speak('What is your starting location?')
            startingPoint = record()
            speak('Where do you want to go?')
            destinationPoint = record()
            try:
                distance = giveDirections(startingPoint, destinationPoint)
                speak("You have to cover a distance of" + distance)
            except Exception as e:
                speak("The location is not proper please try again!")
    elif isContain(query, ['wiki', 'who is', 'wikipedia']):
        speak("Searching")
        result = wikiResult(query)
        speak("According to wikipedia " + result)
    elif isContain(query, ['play']):
        speak("Opening the song")
        youtube_specific(query)
    # elif isContain(query,['weather','current','temperature']):
    #     getWeather(query)

    else:
        res = ''
        tag = replyFunction(query)  # Reply from particular tag
        # print(res)
        if tag == 'time_greeting':
            hour()
        elif tag == 'greeting':
            greeting()
        elif tag == 'create_note':
            create_note()
        elif tag == 'add_todo':
            add_todo()
        elif tag == 'show_todos':
            show_todos()
        elif tag == 'stock':
            getFinance(query)
        elif tag == 'crypto':
            getCrypto(query)
        elif tag == 'weather':
            getWeather(query)
        elif tag == 'news':
            speak('Showing you the latest news')
            news()
        elif tag == 'time':
            time()
        elif tag == 'date':
            date()
        elif tag == 'day':
            current_day()
        elif tag == 'dictionary':
            dictionary(query)
        elif tag == 'question':
            speak("What question do you want to ask?")
            questionQuestion=record()
            qna(questionQuestion)
        elif tag == 'math_operation':
            speak(f"The answer is " + perform(query))
        elif tag == 'exit':
            exitTheApp()
        else:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    print(tag)
                    res = random.choice(intent['responses'])
        output = translator.translate(res, dest='en')
        # print(output)
        if output.pronunciation is not None:
            speak(output.pronunciation)
        else:
            speak(output.text)


def driverProgram():
    try:
        Thread(target=voiceMedium()).start()
    except:
        pass


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r',encoding='utf-8') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    print(all_words)
    # print(len(all_words))
    tags = data['tags']
    model_state = data["model_state"]  # model.state_dict()
    # print(tags)
    # model must be created again with parameters
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    ###GUI SETUP###
    window = Tk()
    window.resizable(width=False, height=False)
    window.configure(width=470, height=700, bg=BG_COLOR)
    window.title('Voice Assistant')

    # head label
    head_label = Label(window, bg=BG_COLOR, fg=TEXT_COLOR,
                       text="Welcome! How can I help you?", font=FONT_BOLD, pady=10)
    head_label.place(relwidth=1)

    # tiny divider
    line = Label(window, width=450, bg=BG_GRAY)
    line.place(relwidth=1, rely=0.07, relheight=0.012)

    frame = Frame(window)
    sc = Scrollbar(frame)
    msgs = Listbox(frame, width=70, height=20)
    sc.pack(side=RIGHT, fill=Y)

    msgs.pack(side=LEFT, fill=BOTH, pady=10)

    frame.pack()

    bottomframe = Frame(window, bg='#dfdfdf', height=100)
    bottomframe.pack(fill=X, side=BOTTOM)

    AITaskStatusLbl = Label(bottomframe, text='Offline', fg='white', bg=AITaskStatusLblBG,
                            font=('montserrat', 16), anchor=CENTER)
    AITaskStatusLbl.place(relx=0.5, rely=0.5)

    # button widget
    b2 = Button(bottomframe, text="Press to Start", command=driverProgram)
    b2.place(relx=0.5, rely=0.5, relwidth=0.5, anchor=CENTER)

    # bottomFrame1 = Frame(window, bg='#dfdfdf')
    # bottomFrame1.pack(fill=X, side=BOTTOM)

    try:
        Thread(target=voiceMedium()).start()
    except:
        pass

    # window.mainloop()
