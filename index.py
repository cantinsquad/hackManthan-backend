import ast
from collections import defaultdict
import json
import os
from flask import Flask, jsonify, url_for, redirect, request, Blueprint
import cv2
import pytesseract
from PIL import Image
from googletrans import Translator
import spacy
import urllib
from string import punctuation
import json
import nltk
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import pymongo
import re

import math

# double rad(double x) {
#     return x * Math.pi / 180;
#   }

#   double getD(LatLng p1, LatLng p2) {
#     double R = 6378137;
#     double dLat = rad(p2.latitude - p1.latitude);
#     double dLong = rad(p2.longitude - p1.longitude);
    # double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    #     Math.cos(rad(p1.latitude)) *
    #         Math.cos(rad(p2.latitude)) *
    #         Math.sin(dLong / 2) *
    #         Math.sin(dLong / 2);
#     double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
#     return R * c / 1000;
#   }



mongoClinet = pymongo.MongoClient('mongodb://127.0.0.1:27017',socketTimeoutMS = 5 * 60000, connectTimeoutMS = 5 * 60000,serverSelectionTimeoutMS = 5 * 60000)

db = mongoClinet['hackManthan']

app = Flask(__name__)

translator = Translator()

def preprocessing(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove shadows, cf. https://stackoverflow.com/a/44752405/11089932
    # dilated_img = cv2.dilate(gray, np.ones((3, 3), np.uint8))
    bg_img = cv2.medianBlur(gray, 25)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Threshold using Otsu's
    return cv2.threshold(norm_img, 0, 255, cv2.THRESH_OTSU)[1]


def getDist(lat1, long1, lat2, long2):

    r = 6378137

    lat = math.radians(lat2 - lat1)    

    long = math.radians(long2 - long1)

    a = math.sin(lat / 2) * math.sin(lat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(long / 2) * math.sin(long / 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


    return r * c / 1000



def wikifier(text, lang="en", threshold=0.8):
    """Function that fetches entity linking results from wikifier.com API"""
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", "jzogyjrcgqsesuayxibnyoizglvodg"),
        ("pageRankSqThreshold", "%g" %
         threshold), ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "100"), ("nWordsToIgnoreFromList", "100"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
        ("includeCosines", "false"), ("maxMentionEntropy", "3")
    ])
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout=60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))
    # Output the annotations.
    results = set()
    for annotation in response["annotations"]:
        # Filter out desired entity classes
        if ('title' in annotation):
          results.add(annotation['title'])



    return results


def azureTextAnalysis(text):
    credential = AzureKeyCredential("KEY")


    text_analytics_client = TextAnalyticsClient(endpoint="URL", credential=credential)


    r = text_analytics_client.recognize_linked_entities([text])

    return r

@app.route('/')
def home_page():
    return 'Hello World'


@app.route('/getReportDetails')
def get_criminals():

    d = json.loads(request.json)

    mongoCollection = db['reports']

    return jsonify(mongoCollection.find({'fir' : d['fir']}))

@app.route('/addReport',methods=['POST'])
def add_report():

    mongoCollection = db['reports']

    d = request.form

    # print(request.files)
    # d = json.load(request.files['data'])

    print(d)

    uid = d["uid"]

    fir = d["fir"]

    lat = float(d["lat"])

    long = float(d["long"])

    print(d)
    print("Hello")

    files = request.files.getlist('images')

    paths = []
    for image in files:
        image_name = image.filename
        image.save(os.path.join(os.getcwd(), image_name))
        paths.append(os.path.join(os.getcwd(), image_name))

    print(paths)

    titles = set()

    for path in paths:
        img = preprocessing(path)

        res = pytesseract.image_to_string(img, config = '-l eng+hin --oem 3 --psm 1')

        x = translator.translate(res, dest = 'en')

        # print(x)
        finalText = re.sub('\s+', ' ', x.text)

        res = azureTextAnalysis(finalText)

        for x in res:
            for y in x['entities']:
                titles.add(y['name'])


    print(titles)

    mongoCollection.insert_one({"fir" : fir, "uid" : uid, "lat" : lat, "long" : long, 'titles' : list(titles) })

    print("Inserted")


    # mongoCollection.update_one({"fir" : fir, "uid" : uid}, {'$push': {'titles': list(titles)}}, upsert = True)

    # print("Pushed")


    allReps = mongoCollection.find()

    reps = defaultdict(lambda : [0,0])

    for report in allReps:
        for t in report["titles"]:
            if t in titles:
                reps[report["fir"]][0] += 1


        
        reps[report["fir"]][1] = getDist(lat,long,report["lat"],report["long"])

    sortedDict = sorted(reps.items(), key = lambda x : x[1][0], reverse = True)
    firs = [y[0] for y in sortedDict]

    dists = [y[1][1] for y in sortedDict]






    return jsonify({'firs' : firs, 'dists' : dists})
#     pass


if __name__ == "__main__":
    app.run('0.0.0.0',5000,True)