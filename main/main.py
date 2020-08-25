import csv
import os
import re
from FileReader import FileDataStructModel as FileDataStruct
from FileReader import FileReaderManager as fileManager
from FeatureExtractManager import FeatureExtractManager as featureManager

personIdRow = 0
genderRow = 3
ageRow = 4
numberRegex = r'\d+'
folder = "/Volumes/HD 2/Mestrado/Aulas Mineração/Base de dados/b5-corpus v.1.7/post/normalised/"
csvFolder = "/Volumes/HD 2/Mestrado/Aulas Mineração/Base de dados/b5-corpus v.1.7/subjects table/subject(PT).csv"
def getData():
    with open(csvFolder, 'r',encoding="iso8859") as file:
        csvFile = list(csv.reader(file,delimiter=";"))
        row_count = sum(1 for row in csvFile)
        data = {}
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                personId = re.findall(numberRegex, filename)[0]
                age = 0
                gender = ""
                i = 0
                while i < row_count:
                    row = csvFile[i]
                    if personId == row[personIdRow]:
                        age = row[ageRow]
                        gender = row[genderRow]
                        adjCount = featureManager.ponctuationNumber(fileManager.getText(folder + filename, encoding="iso8859"))
                        struct = FileDataStruct.FileDataStruct("", age, gender,adjCount)
                        data[int(personId)] = struct
                    i = i + 1
    return data