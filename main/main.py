import csv
import os
import re
from FileReader import FileDataStructModel as FileDataStruct
from FileReader import FileReaderManager as fileManager
from FeatureExtractManager import FeatureExtractManager as featureManager
from datetime import date

personIdRow = 0
genderRow = 3
ageRow = 4
numberRegex = r'\d+'
folder = "/Volumes/HD 2/Mestrado/Aulas Mineração/Base de dados/b5-corpus v.1.7/post/normalised/"
csvFolder = "/Volumes/HD 2/Mestrado/Aulas Mineração/Base de dados/b5-corpus v.1.7/subjects table/subject(PT).csv"
csvReviewFolder = "/Volumes/HD 2/Mestrado/Aulas Mineração/Base de dados/b2w-reviews01-master/B2W-Reviews01.csv"
def getData():
    with open(csvFolder, 'r',encoding="iso8859") as file:
        csvFile = list(csv.reader(file,delimiter=";"))
        row_count = sum(1 for row in csvFile)
        data = {}
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                personId = re.findall(numberRegex, filename)[0]
                idade = 0
                gender = ""
                i = 0
                while i < row_count:
                    row = csvFile[i]
                    if personId == row[personIdRow]:
                        if (row[ageRow] != ""):
                            a = int(row[ageRow])
                            if a >= 10 and a <= 20:
                                idade = 0
                            elif a > 20 and a <= 30:
                                idade = 1
                            elif a > 30 and a <= 40:
                                idade = 2
                            if row[genderRow] == "male":
                                gender = 0
                            else:
                                gender = 1
                            text = fileManager.getText(folder + filename, encoding="iso8859")
                            grams = featureManager.grams(text)
                            #postag = featureManager.posTagCount(text)
                            struct = FileDataStruct.FileDataStruct("", idade, gender,"",grams)
                            data[int(personId)] = struct
                    i = i + 1
    return data

def getReviewData():
    with open(csvReviewFolder, 'r',encoding="utf-8") as file:
        csvFile = list(csv.reader(file,delimiter=";"))
        row_count = sum(1 for row in csvFile)
        data = {}
        Pass = 0
        i = 0
        idade = 0
        for values in csvFile:
            if values[12] == "null" or values[11] == "null" or values[10] == "null":
                continue
            Pass += 1
            if Pass > 1:
                a = int(date.today().year - int(values[11]))
                if a <= 60 and a >= 10:
                    if a >= 20 and a <= 30:
                        idade = 0
                    elif a > 30 and a <= 40:
                        idade = 1
                    elif a > 40:
                        idade = 2
                    if values[12] == "M":
                        gender = 0
                    else:
                        gender = 1
                    text = values[10]
                    grams = featureManager.grams(text)
                    # postag = featureManager.posTagCount(text)
                    struct = FileDataStruct.FileDataStruct("", idade, gender, "", grams)
                    data[i] = struct
                    i = i + 1
    return data