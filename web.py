import math
import webbrowser
import wikipedia
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
import nltk
import time
import requests, json

from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from geopy.distance import great_circle

from speech import *

stopword = nltk.corpus.stopwords.words("english")
custom_stopwords = ['price', 'today', 'please', 'search', 'web', 'order', 'weather', 'temperature', 'today', 'give',
                    'meaning', 'calculate', 'value']
for word in custom_stopwords:
    stopword.append(word)
# print(stopword)

finances = {
    'sensex': 'SENSEX:INDEXBOM',
    'nifty': 'NIFTY_50:INDEXNSE',
    'hsi': 'HSI:INDEXHANGSENG',
    'sse': '000001:SHA',
    'nikkei': 'NI225:INDEXNIKKEI',
    'nasdaq': '.IXIC:INDEXNASDAQ'
}
crytos = {
    'bitcoin': 'BTC-INR',
    'ethereum': 'ETH-INR',
    'cardano': 'ADA-INR',
    'xrp': 'XRP-INR',
    'dogecoin': 'DOGE-INR'
}


# print(stopword)


# print(finance.keys())
# if 'nifty' in finance.keys():
#     print(finance['nifty'])
def getCrypto(k):
    argument = ''
    k = k.lower()
    list = re.split("\W+", k)
    text = [word for word in list if word not in stopword]
    if 'price' in text:
        text.remove('price')
    keyword = ' '.join(map(str, text))
    if keyword in crytos.keys():
        argument = crytos[keyword]
    driver = webdriver.Chrome('C:/Users/Anu-PC/Downloads/chromedriver_win32/chromedriver.exe')
    driver.get("https://www.google.com/finance/quote/" + argument)
    time.sleep(10)
    driver.quit()


def getFinance(k):
    argument = ''
    k = k.lower()
    list = re.split("\W+", k)
    text = [word for word in list if word not in stopword]
    if 'today' in text:
        text.remove('today')
    keyword = ' '.join(map(str, text))
    if keyword in finances.keys():
        argument = finances[keyword]
    driver = webdriver.Chrome('C:/Users/Anu-PC/Downloads/chromedriver_win32/chromedriver.exe')
    driver.get("https://www.google.com/finance/quote/" + argument)
    time.sleep(10)
    driver.quit()


def web(k):
    list = re.split("\W+", k)
    text = [word for word in list if word not in stopword]
    print(text)
    if 'search' in text:
        text.remove("search")
    if 'web' in text:
        text.remove("web")
    keyword = ' '.join(map(str, text))
    driver = webdriver.Chrome('C:/Users/Anu-PC/Downloads/chromedriver_win32/chromedriver.exe')
    driver.get("https://google.co.in/search?q=" + keyword)
    time.sleep(100)
    driver.quit()


def youtube(k):
    list = re.split("\W+", k)
    text = [word for word in list if word not in stopword]
    print(text)
    if 'search' in text:
        text.remove("search")
    text.remove("youtube")
    keyword = ' '.join(map(str, text))
    driver = webdriver.Chrome('C:/Users/Anu-PC/Downloads/chromedriver_win32/chromedriver.exe')
    driver.get("https://www.youtube.com/results?search_query=" + keyword)
    time.sleep(10)
    driver.quit()


def flipkart(query):
    list = re.split("\W+", query)
    text = [word for word in list if word not in stopword]
    driver = webdriver.Chrome('C:/Users/Anu-PC/Downloads/chromedriver_win32/chromedriver.exe')
    if 'order' in text:
        text.remove("order")

    text.remove("flipkart")
    keyword = ' '.join(map(str, text))
    driver.get("https://www.flipkart.com/search?q=" + keyword)
    time.sleep(10)
    driver.quit()


def amazon(query):
    list = re.split("\W+", query)
    text = [word for word in list if word not in stopword]
    driver = webdriver.Chrome('C:/Users/Anu-PC/Downloads/chromedriver_win32/chromedriver.exe')

    if 'order' in text:
        text.remove("order")
    text.remove("amazon")
    keyword = ' '.join(map(str, text))
    driver.get("https://www.amazon.in/s?k=" + keyword)
    time.sleep(10)
    driver.quit()
    # https://www.amazon.in/s?k=


def news():
    driver = webdriver.Chrome('C:/Users/Anu-PC/Downloads/chromedriver_win32/chromedriver.exe')
    driver.get("https://news.google.com/topstories?hl=en-IN&gl=IN&ceid=IN:en")
    time.sleep(5)


def getWeather(query):
    api_key = "9c233a297d8e156ddcc712d80e53c749"

    # base_url variable to store url
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    list = re.split("\W+", query)
    text = [word for word in list if word not in stopword]
    if 'weather' in text:
        text.remove('weather')
    if 'current' in text:
        text.remove('weather')
    if 'temperature' in text:
        text.remove('temperature')
    if 'today' in text:
        text.remove('today')

    cityname = ' '.join(map(str, text))

    complete_url = base_url + "appid=" + api_key + "&q=" + cityname

    # get method of requests module
    # return response object
    response = requests.get(complete_url)

    # json method of response object
    # convert json format data into
    # python format data
    x = response.json()

    if x["cod"] != "404":

        # store the value of "main"
        # key in variable y
        y = x["main"]

        # store the value corresponding
        # to the "temp" key of y
        current_temperature = y["temp"]

        # store the value corresponding
        # to the "humidity" key of y
        current_humidity = y["humidity"]

        # store the value of "weather"
        # key in variable z
        z = x["weather"]

        # store the value corresponding
        # to the "description" key at
        # the 0th index of z
        weather_description = z[0]["description"]

        speak(
            f"The current temperature of {cityname} is {math.ceil(current_temperature - 273)} degree celsius with {weather_description} weather")

    else:
        speak(" City Not Found ")


def jokes():
    URL = 'https://icanhazdadjoke.com/'
    result = requests.get(URL)
    src = result.content

    soup = BeautifulSoup(src, 'html.parser')

    try:
        p = soup.find('p')
        speak(p.text)
    except Exception as e:
        raise e


def openWebsite(query):
    list = re.split("\W+", query)
    appendedQuery = ''
    for word in list:
        appendedQuery += word + "+"
    appendedQuery = appendedQuery[:-1]
    webbrowser.open(f"https://www.google.com/search?q={appendedQuery}")


def giveDirections(startingPoint, destinationPoint):
    geolocator = Nominatim(user_agent='assistant')
    if 'current' in startingPoint:
        res = requests.get("https://ipinfo.io/")
        data = res.json()
        startinglocation = geolocator.reverse(data['loc'])
    else:
        startinglocation = geolocator.geocode(startingPoint)

    destinationlocation = geolocator.geocode(destinationPoint)
    startingPoint = startinglocation.address.replace(' ', '+')
    destinationPoint = destinationlocation.address.replace(' ', '+')

    openWebsite('https://www.google.co.in/maps/dir/' + startingPoint + '/' + destinationPoint + '/')

    startinglocationCoordinate = (startinglocation.latitude, startinglocation.longitude)
    destinationlocationCoordinate = (destinationlocation.latitude, destinationlocation.longitude)
    total_distance = great_circle(startinglocationCoordinate, destinationlocationCoordinate).km  # .mile
    return str(round(total_distance, 2)) + 'Kilometer'


def wikiResult(query):
    query = query.replace('wikipedia', '')
    query = query.replace('search', '')
    if len(query.split()) == 0:
        query = "wikipedia"
    try:

        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return "Desired Result Not Found"


def youtube_specific(query):
    list = re.split("\W+", query)
    list.remove('play')
    search_query = ''
    for word in list:
        search_query = search_query + '+' + word
    search_query = search_query[1:]
    # print(search_query)
    # print(f"https://www.youtube.com/results?search_query={search_query}")
    response = requests.get(f"https://www.youtube.com/results?search_query={search_query}").text
    # print(response)
    soup = BeautifulSoup(response, 'lxml')
    # print(soup)
    script = soup.find_all("script")[33]
    # print(script)
    try:
        json_text = re.search('var ytInitialData = (.+)[.;]{1}', str(script)).group(1)
    except:
        json_text = re.search('var ytInitialData = (.+)[.;]{1}', str(script))

    json_data = json.loads(json_text)

    content = (
        json_data['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents'][
            0][
            'itemSectionRenderer']['contents'])
    videoIdList = []
    for data in content:
        for key, value in data.items():
            if type(value) is dict:
                for k, v in value.items():
                    if k == 'videoId' and len(v) == 11:
                        videoIdList.append(v)
    # global driver
    # chrome_options = Options()
    # chrome_options.add_experimental_option("detach", True)
    # driver = webdriver.Chrome('C:/Users/Anu-PC/Downloads/chromedriver_win32/chromedriver.exe')
    #
    # driver.get(f"https://www.youtube.com/watch?v={videoIdList[0]}?autoplay=1")
    webbrowser.open(f"https://www.youtube.com/watch?v={videoIdList[0]}?autoplay=1")


# youtube_specific('play justin beiber what do you mean')

def sendWhatsapp(phone_no='', message=''):
    phone_no = '+91' + str(phone_no)
    webbrowser.open('https://web.whatsapp.com/send?phone=' + phone_no + '&text=' + message)
    import time
    from pynput.keyboard import Key, Controller
    time.sleep(5)
    k = Controller()
    k.press(Key.enter)

# youtube_specific('play despacito')
