import sys
import argparse
import main as m
import pandas as pd
import GenderPredictionManager as genderManager
import FeatureExtractManager as featureManager
from pathlib import Path


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Args for each of the models')
    parser.add_argument('-m',
                        help="""The model (1-B5-Corpus, 2-B2W-Reviews,
                        3-Blogset-BR, 4-PAN, 5-ESIC, 6-BRMORAL)
                        """,
                        type=int,
                        dest="model")
    parser.add_argument('-c',
                        help='Uses cascade model',
                        type=bool,
                        dest="cascade")

    args = parser.parse_args()

    main(args)

def main(params):
    model = params.model
    corpusName = f'corpus-{model}'

    if (Path(corpusName).is_file() == False):
        if model == 1:
            data = m.getData()
        elif model == 2:
            data = m.getReviewData()
        elif model == 3:
            data = m.getBlogsetData()
        elif model == 4:
            data = m.getAllPan()
        elif model == 5:
            data = m.getESic()
        elif model == 6:
            data = m.getBrMoral()
        else:
            input("Invalid option. Please see help with -h argument")

        id = []
        gender = []
        text = []
        predictedGender = []
        for entry in data.keys():
            id.append(entry)
            gender.append(data[entry].gender)
            text.append(data[entry].text)
            predictedGender.append(data[entry].predictedGender)
        data = {}
        data["id"] = id
        data["gender"] = gender
        data["predictedGender"] = predictedGender
        data["text"] = text
        originalDF = pd.DataFrame(data, columns=["id", 'gender',
                                            "predictedGender","text"])
        originalDF.to_csv(corpusName, index=True)
    else:
        originalDF = pd.read_csv(corpusName)

    if predictGenders == "SIM":
        genderManager.printPredictedGenderScores(originalDF)
        quit()

    genderHeuristic = False
    if chooseClassification != 3:
        if chooseHeuristica == "SIM":
            genderHeuristic = True
        else:
            genderHeuristic = False

    if chooseClassification == "1":
        genderManager.b5CorpusPrediction(predictAttribute=originalDF.gender,
                                        dataFrame=originalDF,
                                        genderHeuristica=genderHeuristic)
    elif chooseClassification == "2":
        genderManager.panCorpusPrediction(originalDF,genderHeuristic)
    elif chooseClassification == "3":
        originalDF = originalDF.dropna()
        originalDF = originalDF.astype({"idade": int})
        originalDF = originalDF.reset_index(drop=True)
        genderManager.b5CorpusPrediction(predictAttribute=originalDF.idade,
                                        dataFrame=originalDF,
                                        genderHeuristica=False)
    elif chooseClassification == "4":
        if choosedDataSet == "3": # Blogset BR
            genderManager.crossDomainPrediction(originalDF,w=600,
                                                s="pre",x=600,
                                                filter=False,
                                                it=200,
                                                layers=(300),
                                                f="relu",
                                                alpha=1e-05,
                                                corpusName="blog",
                                                corpusThreshold=0.77,
                                                genderHeuristica=genderHeuristic)
        elif choosedDataSet == "6": # E-Gov
            genderManager.crossDomainPrediction(originalDF,
                                                w=1000,
                                                s="pre",
                                                x=1000,
                                                filter=True,
                                                it=200,
                                                layers=(500),
                                                f="relu",
                                                alpha=1e-05,
                                                corpusName="e-gov",
                                                corpusThreshold=0.78,
                                                genderHeuristica=genderHeuristic)
        else: # Opinion
            genderManager.crossDomainPrediction(originalDF,
                                                w=100,
                                                s="self",
                                                x=80,
                                                filter=True,
                                                it=250,
                                                layers=(25,25,25),
                                                f="tanh",
                                                alpha=1e-07,
                                                corpusName="opinion",
                                                corpusThreshold=0.74,
                                                genderHeuristica=genderHeuristic)


