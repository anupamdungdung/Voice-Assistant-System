import pyttsx3
import datetime
import pyjokes

engine = pyttsx3.init()
voices = engine.getProperty('voices')
rate = engine.getProperty("rate")

# engine.setProperty("voice", voices[20].id) # For Hindi Voice
engine.setProperty("voice", voices[1].id)  # For English Voice
engine.setProperty("rate", 135)


def speak(audio):
    try:
        engine.say(audio)
        engine.runAndWait()
    except Exception as e:
        print(e)


def wishme():
    # speak("Welcome back!")
    # speak("Today's date is")
    # date()
    speak("Hello. How can I help you?")


def date():
    year = int(datetime.datetime.now().year)
    month = int(datetime.datetime.now().month)
    date = int(datetime.datetime.now().day)
    speak(f"Today's date is {date} {month} {year}")
    # speak(date)
    # speak(month)
    # speak(year)


def time():
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    if 1 <= hour < 12:
        timeOfDay = 'am'
    else:
        timeOfDay = 'pm'
    speak(f'Current time is {hour - 12}{minute}{timeOfDay}')

def current_day():
    day=datetime.datetime.now()
    speak(f'Today is {day.strftime("%A")}')
