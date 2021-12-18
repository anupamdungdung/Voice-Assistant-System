import speech_recognition as sr
from playsound import playsound
from speech import *

r = sr.Recognizer()
r.energy_threshold = 2000

todo_list = []


def create_note():
    global r
    speak('What do you want to write onto your to do list?')

    done = False

    while not done:
        try:
            with sr.Microphone() as mic:
                print("Listening........")
                audio = r.listen(mic)

                note = r.recognize_google(audio)
                note = note.lower()

                speak('Choose a filename')

                print("Listening........")
                audio = r.listen(mic)

                filename = r.recognize_google(audio)
                filename = filename.lower()

            with open(f'{filename}.txt', 'w') as f:
                f.write(note)
                done = True
                speak(f'I successfully added the note {filename}')

        except sr.UnknownValueError:
            speak('I could not understand! Please say the task again! ')


def add_todo():
    global r

    speak('What to do do you want to add?')

    done = False

    while not done:
        try:
            with sr.Microphone() as mic:
                print("Listening........")
                audio = r.listen(mic)

                item = r.recognize_google(audio)
                item = item.lower()

                todo_list.append(item)
                done = True

                speak(f'I added {item} to the to do list')

        except sr.UnknownValueError:
            speak('I could not understand! Please try again! ')


def show_todos():
    if len(todo_list) == 0:
        speak('There are no items on your to do list')
    else:
        speak('The items on the to do list are')
        for item in todo_list:
            speak(item)


def happyBirthday():
    playsound("D:/Skill Development/Speech Recognition/happybirthday.mp3")


# happyBirthday()


