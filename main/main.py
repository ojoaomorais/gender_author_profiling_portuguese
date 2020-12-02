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
        print("CSV Total: %s" % len(csvFile))
        row_count = sum(1 for row in csvFile)
        data = {}
        errors = 0
        for filename in os.listdir(folder):
            personId = re.findall(numberRegex, filename)[0]
            idade = 0
            gender = ""
            i = 0
            result = list(filter(lambda x: x[0] == personId, csvFile))
            try:
                a = int(result[0][ageRow])
                if a >= 10 and a < 20:
                    idade = 0
                elif a >= 20 and a < 30:
                    idade = 1
                elif a >= 30 and a <= 61:
                    idade = 2
                if result[0][genderRow] == "male":
                    gender = 0
                else:
                    gender = 1
                text = fileManager.getText(folder + filename, encoding="iso8859")
                struct = getFileStruct(idade, gender, text,id=personId)
                data[int(personId)] = struct
            except ValueError:
                errors = errors + 1
                #print("Errors %s" %errors)
                pass
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
                    if a >= 15 and a <= 30:
                        idade = 0
                    elif a > 30 and a <= 45:
                        idade = 1
                    elif a > 45:
                        idade = 2
                    if values[12] == "M":
                        gender = 0
                    else:
                        gender = 1
                    text = values[10]
                    data[i] = getFileStruct(idade,gender,text,id=values[1])
                    i = i + 1
    return data

def getFileStruct(idade,gender,text,id):
    #grams = featureManager.grams(text,3)
    predictGender = featureManager.getGender(text,gender,id)
    #grams2 = featureManager.grams(text,2)
    #postag = featureManager.posTagCount(text)
    #blankSpaces = featureManager.getBlankSpaces(text)
    #capitalizedCount = featureManager.getCapitalizedCount(text)
    #ponctCount = featureManager.getPontcCount(text)
    #words = featureManager.getWords(text)
    return FileDataStruct.FileDataStruct(text, idade, gender, predictedGender=predictGender)