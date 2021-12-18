import os
import subprocess
import time
import re
from speech import *


def openCalculator(query):
    list = re.split("\W+", query)
    text = [word for word in list]
    if 'open' in text:
        try:
            subprocess.Popen('C:/Windows/System32/calc.exe')
            # time.sleep(5)
        except Exception as e:
            speak("Error! Could not open calculator")
            print(str(e))


def openNotePad(query):
    list = re.split("\W+", query)
    text = [word for word in list]
    if 'open' in text:
        try:
            subprocess.Popen('C:/Windows/System32/notepad.exe')
            # time.sleep(5)

        except Exception as e:
            speak("Error! Could not open Notepad")

            print(str(e))


def closeNotePad():
    try:
        subprocess.call(["taskkill", "/F", "/IM", "notepad.exe"])

    except Exception as e:
        speak('Error! Could not close Notepad')
        print(str(e))


def openCamera():
    try:
        subprocess.run('start microsoft.windows.camera:', shell=True)
        # time.sleep(5)

    except Exception as e:
        speak('Error! Could not open camera')
        print(str(e))


def closeCamera():
    try:
        subprocess.run('Taskkill /IM WindowsCamera.exe /F', shell=True)

    except Exception as e:
        speak('Error! Could not open camera')
        print(str(e))

# def openMSWord():
#     try:
#         os.system('winword')
#     except Exception as e:
#         speak('Error! Could not open MS Word')
#         print(str(e))
