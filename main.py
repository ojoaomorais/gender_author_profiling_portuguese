import csv
import os
import re

from tqdm import tqdm

import FileDataStructModel as FileDataStruct
import FileReaderManager as fileManager
import FeatureExtractManager as featureManager
from datetime import date
import xml.etree.ElementTree as ET
import sys
import pandas as pd
csv.field_size_limit(sys.maxsize)
personIdRow = 0
genderRow = 3
ageRow = 4
numberRegex = r'\d+'
folder = "b5-corpus v.1.7/post/normalised/"
csvFolder = "b5-corpus v.1.7/subjects table/subject(PT).csv"
csvReviewFolder = "B2W-Reviews01.csv"
csvBlogsetFolder = "blogset-br/blogset-br.csv"
csvBlogsetDataFolder = "blogset-br/4300Answers.csv"
csvStilingueFolder = "RelatorioDePublicacoes-_16_Nov_2020a17_Nov_2020.csv"
panFolderTest = "pan17-test"
panFolderTraining = "pan17-training"
eSicFolder = "e-sic.csv"
brMoralFolder = "brMoral.csv"
parsedBlogsetFolder = "parsedBlogset2.csv"
def getData():
    femaleCount = 0
    maleCount = 0
    with open(csvFolder, 'r',encoding="iso8859") as file:
        csvFile = list(csv.reader(file,delimiter=";"))
        print("CSV Total: %s" % len(csvFile))
        row_count = sum(1 for row in csvFile)
        data = {}
        errors = 0
        idade0 = 0
        idade1 = 0
        idade2 = 0
        for filename in os.listdir(folder):
            personId = re.findall(numberRegex, filename)[0]
            idade = None
            gender = None
            i = 0
            result = list(filter(lambda x: x[0] == personId, csvFile))
            a = 666
            if result[0][ageRow] != "":
                a = int(result[0][ageRow])
            if a >= 18 and a <= 20:
                idade = int(0)
                idade0 += 1
            elif a >= 23 and a <= 25:
                idade = int(1)
                idade1 += 1
            elif a >= 28 and a <= 61:
                idade = int(2)
                idade2 += 1
            if result[0][genderRow] == "male":
                gender = 0
                maleCount += 1
            elif result[0][genderRow] == "female":
                femaleCount += 1
                gender = 1
            if gender is not None:
                text = fileManager.getText(folder + filename, encoding="iso8859")
                struct = getFileStruct(idade, gender, text,id=personId)
                data[int(personId)] = struct
        print("Idade 0 :",idade0)
        print("Idade 1 :", idade1)
        print("Idade 2 :", idade2)
        print("Numero Females:",femaleCount)
        print("Numero Males:", maleCount)
        return data

def getReviewData():
    with open(csvReviewFolder, 'r',encoding="utf-8") as file:
        csvFile = list(csv.reader(file,delimiter=";"))
        row_count = sum(1 for row in csvFile)
        data = {}
        Pass = 0
        i = 0
        idade = None
        idade0 = 0
        idade1 = 0
        idade2 = 0
        maleGender = 0
        femaleGender = 0
        for values in csvFile:
            idade = None
            if values[12] == "null" or values[11] == "null" or values[10] == "null":
                continue
            Pass += 1
            if Pass > 1:
                a = int(date.today().year - int(values[11]))
                if a != "":
                    if a >= 10 and a < 31:
                        idade = "10 a 30"
                    elif a >= 31 and a < 40:
                        idade = "31 a 40"
                    elif a >= 41 and a < 50:
                        idade = "41 a 50"
                    elif a >= 51 and a < 60:
                        idade = "51 a 60"
                    else:
                        idade = "Maior que 60"
                if values[12] == "M":
                    gender = 0
                    maleGender += 1
                else:
                    gender = 1
                    femaleGender += 1
                text = values[10]
                data[i] = getFileStruct(idade,gender,text,id=values[1])
                i = i + 1
    print("Quantidade Masc",maleGender)
    print("Quantidade Fem",femaleGender)
    return data

def getStilingueData():
    with open(csvStilingueFolder, 'r',encoding="utf-8") as file:
        csvFile = list(csv.reader(file,delimiter=","))
        row_count = sum(1 for row in csvFile)
        data = {}
        Pass = 0
        i = 0
        idade = 0
        for values in csvFile:
            Pass += 1
            if Pass > 1:
                #a = int(date.today().year - int(values[11]))
                #if a <= 60 and a >= 10:
                #if a >= 15 and a <= 30:
                #idade = 0
                #elif a > 30 and a <= 45:
                #idade = 1
                #elif a > 45:
                #idade = 2
                if values[14] == "Homem":
                    gender = 0
                else:
                    gender = 1
                text = values[11]
                data[i] = getFileStruct(0,gender,text,id=values[12])
                i = i + 1
    return data

def getAllPan():
    test = getPan(panFolderTest,False)
    training = getPan(panFolderTraining,False)
    d = test.copy()
    d.update(training)
    return d

def getPan(path,onlyBrazil):
    data = {}
    maleCount = 0
    femaleCount = 0
    for folder in os.listdir(path):
        if not folder.endswith(".DS_Store") and (folder == "pt"):
            folderFiles = path + "/" + folder
            truth = folderFiles + "/truth.txt"
            with open(truth, 'r', encoding="iso8859") as file:
                fileId = ""
                fileIds = [line.rstrip() for line in file]
                files = os.listdir(folderFiles)
                for filename in files:
                    gender = 666
                    genderPattern = [fid for fid in fileIds if fid.startswith(filename.partition(".")[0])]
                    if filename.endswith("xml") and len(genderPattern) > 0:
                        head, sep, tail = filename.partition('.')
                        fileId = head
                        tree = ET.parse(folderFiles + "/" + filename)
                        root = tree.getroot()
                        text = ""
                        for elem in root.iter():
                            if elem.tag == "document":
                                cleaned = re.sub('(.+?) — ', '', elem.text)
                                text += ". " + cleaned
                        h, s, t = genderPattern[0].partition(':::')
                        t1,t2,t3 = t.partition(":::")
                        if onlyBrazil:
                            if t3 == "brazil":
                                if t1 == "male":
                                    gender = 0
                                    maleCount += 1
                                else:
                                    gender = 1
                                    femaleCount += 1
                        else:
                            if t1 == "male":
                                gender = 0
                                maleCount += 1
                            else:
                                gender = 1
                                femaleCount +=1
                        struct = getFileStruct(666, gender, text, id=fileId)
                        data[fileId] = struct
    print("Quantidade Masc", maleCount)
    print("Quantidade Fem", femaleCount)
    return data

def generateBlogsetCSV():
    with open(csvBlogsetFolder, 'r', encoding="utf-8") as texts:
        with open(csvBlogsetDataFolder, 'r', encoding="utf-8") as answers:
            textsCSV = list(csv.reader(texts, delimiter=","))
            answerCSV = list(csv.reader(answers, delimiter=","))
            data = {}
            Pass = 0
            i = 0
            idade = 0
            qtde = 0
            tweets = []
            genders = []
            userId = []
            maleCount = 0
            femaleCount = 0
            gender = None
            for values in tqdm(answerCSV):
                model = [x for i, x in enumerate(textsCSV) if x and values[1] == x[5]]
                if model:
                    text = ""
                    if (values[5] == "Masculino"):
                        gender = 0
                        maleCount += 1
                    else:
                        gender = 1
                        femaleCount += 1
                    for m in model:
                        text += m[4] + "."
                    tweets.append(text)
                    genders.append(gender)
                    userId.append(values[1])
            twiData = {}
            twiData["user_id"] = userId
            twiData["gender"] = genders
            twiData["text"] = tweets
            twiDF = pd.DataFrame(twiData, columns=["user_id", 'gender', 'text'])
            twiDF.to_csv("parsedBlogset2.csv", sep='\t', encoding='utf-8')

def getBlogsetData():
    with open(parsedBlogsetFolder, 'r', encoding="utf-8") as file:
        dataCsvFile = list(csv.reader(file, delimiter="\t"))
        data = {}
        Pass = 0
        i = 0
        idade = 0
        qtde = 0
        maleCount = 0
        femaleCount = 0
        for values in tqdm(dataCsvFile):
            Pass += 1
            if Pass > 1:
                if int(values[2]) == 0:
                    gender = 0
                    maleCount += 1
                else:
                    gender = 1
                    femaleCount += 1
                text = values[3]
                data[i] = getFileStruct(0, gender,text, id=id)
                i = i + 1
    print("Qtde encontrada:", qtde)
    print("Quantidade Masc", maleCount)
    print("Quantidade Fem", femaleCount)
    return data

def getFileStruct(idade,gender,text,id):
    predictGender = featureManager.getGenderStanza(text,gender)
    #emojiFrequency = featureManager.getEmojiFrequency(text)
    #laughFrequency = featureManager.getLaughFrequency(text)
    #slangFrequency = featureManager.getSlangFrequency(text)

    return FileDataStruct.FileDataStruct(text=text,
                                         idade= idade,
                                         gender= gender,
                                         predictedGender=predictGender,
                                         slangFrequency=0,
                                         laughFrequency=0,
                                         emojiFrequency=0
                                         )

def getESic():
    with open(eSicFolder, 'r',encoding="ISO-8859-1") as file:
        csvFile = list(csv.reader(file,delimiter=";"))
        row_count = sum(1 for row in csvFile)
        data = {}
        Pass = 0
        i = 0
        maleGender = 0
        femaleGender = 0
        for values in csvFile:
            idade = None
            if values[1] == "null" or values[4] == "null" :
                continue
            Pass += 1
            if Pass > 1:
                if values[1] != "na":
                    if values[1] == "m":
                        gender = 0
                        maleGender += 1
                    else:
                        gender = 1
                        femaleGender += 1
                    text = values[4]
                    text = text.lower()
                    regex = re.compile('[^A-Za-zéêíîôóõáàâãúûçÉÊÍÎÔÓÕÁÀÂÃÚÛÇ]+')
                    text = re.sub(regex, ' ', text)
                    data[i] = getFileStruct(idade,gender,text,id=values[1])
                i = i + 1
    print("Quantidade Masc",maleGender)
    print("Quantidade Fem",femaleGender)
    return data

def getBrMoral():
    with open(brMoralFolder, 'r',encoding="ISO-8859-1") as file:
        csvFile = list(csv.reader(file,delimiter=";"))
        row_count = sum(1 for row in csvFile)
        data = {}
        Pass = 0
        i = 0
        maleGender = 0
        femaleGender = 0
        for values in csvFile:
            text = ""
            idade = None
            if values[12] == "null" or values[12] is None :
                continue
            Pass += 1
            if Pass > 1:
                if values[12] == "m":
                    print(values[12])
                    gender = 0
                    maleGender += 1
                else:
                    print(values[12])
                    gender = 1
                    femaleGender += 1
                for k in range(1,10):
                    text = text + values[k] + "."
                data[i] = getFileStruct(idade,gender,text,id=values[0])
            i = i + 1
    print("Quantidade Masc",maleGender)
    print("Quantidade Fem",femaleGender)
    return data