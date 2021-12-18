import os
import subprocess
from brain_2 import speak

if not os.path.exists('Files and Document'):
    os.mkdir('Files and Document')
path = 'Files and Document/'


def isContain(text, list):
    for word in list:
        if word in text:
            return True
    return False


def createFile(text):
    appLocation = "C:/Users/Anu-PC/AppData/Local/Programs/Microsoft VS Code/Code.exe"

    if isContain(text, ["ppt", "power point", "powerpoint"]):
        file_name = "sample_file.ppt"
        appLocation = "C:/Program Files/Microsoft Office/root/Office16/POWERPNT.exe"

    elif isContain(text, ['excel', 'spreadsheet']):
        file_name = "sample_file.xsl"
        appLocation = "C:/Program Files/Microsoft Office/root/Office16/EXCEL.exe"

    elif isContain(text, ['word', 'document']):
        file_name = "sample_file.docx"
        appLocation = "C:/Program Files/Microsoft Office/root/Office16/WINWORD.exe"

    elif isContain(text, ["text", "simple", "normal"]):
        file_name = "sample_file.txt"
    elif "python" in text:
        file_name = "sample_file.py"
    elif "css" in text:
        file_name = "sample_file.css"
    elif "javascript" in text:
        file_name = "sample_file.js"
    elif "html" in text:
        file_name = "sample_file.html"
    elif "c plus plus" in text or "c + +" in text:
        file_name = "sample_file.cpp"
    elif "java" in text:
        file_name = "sample_file.java"
    elif "json" in text:
        file_name = "sample_file.json"
    else:
        speak("Unable to create this type of file")

    file = open(path + file_name, 'w')
    file.close()
    subprocess.Popen([appLocation, path + file_name])
    speak("File is created. Now you can edit this file")
