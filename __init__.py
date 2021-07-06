import sys

from tqdm import tqdm
from sklearn.multioutput import ClassifierChain
import main as m
import pandas as pd
import nltk
import GenderPredictionManager as genderManager
import FeatureExtractManager as featureManager

def classifierChain(dataFrame):
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.svm import SVC
    X = originalDF[['laughFrequency', "emojiFrequency", "slangFrequency", "predictedGender"]]
    Y = originalDF[['idade', 'gender']]
    # Binarizando coluna idade
    Y = pd.concat([originalDF.gender, pd.get_dummies(originalDF.idade)], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,
                                                        random_state=0)

    base_lr = SVC()
    chain = ClassifierChain(base_lr, order=[0, 1, 2, 3], random_state=0)

    from sklearn import metrics
    scores = cross_val_score(chain, X, Y, cv=5, scoring='f1_macro')
    print(scores)



# import tweepy
# import json
# tweets = []
# genders = []
# userId = []
# auth = tweepy.OAuthHandler("pmT2pVdObh5ThqWviO0ivXqPv", "mUfZbzU1ZNtCYHEjlCj8Yihu28xYI80MABNpbz5dprVGCPxskV")
# auth.set_access_token("3026375668-LCgRvotj9nVTl82nvWGX0utJrz6IlBzHx3xuwUY", "gQX2c4F6QpZM3NWMpCqJWgnKQNZFtQoM7inHgtQFHBbNP")
# api = tweepy.API(auth, wait_on_rate_limit=True)
# with open("twisty/TwiSty-PT.json", 'r',encoding="utf-8") as file:
#     y = json.load(file)
#     for d in  tqdm(y):
#         #for twi in y[d]["other_tweet_ids"]:
#         #    try:
#         #        tweet = api.get_status(twi)
#         #        print(tweet.text)
#         #        tweets.append(twi)
#         #        genders.append(y[d]["gender"])
#         #        userId.append(y[d]["user_id"])
#         #    except tweepy.TweepError:
#         #        print("Error")
#         for twi in tqdm(y[d]["confirmed_tweet_ids"]):
#             try:
#                 tweet = api.get_status(twi)
#                 tweets.append(twi)
#                 genders.append(y[d]["gender"])
#                 userId.append(y[d]["user_id"])
#             except tweepy.TweepError as e:
#                 pass
#     twiData = {}
#     twiData["user_id"] = userId
#     twiData["gender"] = genders
#     twiData["text"] = tweets
#     twiDF = pd.DataFrame(twiData, columns=["user_id", 'gender', 'text'])
#
#     twiDF.to_csv("twisty.csv", sep='\t', encoding='utf-8')

# average = input("Média de palavras por documento")
#
# if average == "SIM":
#     datasets = []
#     datasetsName = ['## 1 - b5-corpus','## 2 - B2W reviews','## 3 - Blogset BR','## 5 - PAN','## 6 - E-Sic','## 7 - Br Moral']
#     datasets.append(m.getData())
#     datasets.append(m.getReviewData())
#     datasets.append(m.getBlogsetData())
#     datasets.append(m.getAllPan())
#     datasets.append(m.getESic())
#     datasets.append(m.getBrMoral())
#     i = 0
#     for dataset in datasets:
#         print("====================================")
#         id = []
#         idade = []
#         gunningIndex = []
#         gender = []
#         text = []
#         predictedGender = []
#         laughFrequency = []
#         emojiFrequency = []
#         slangFrequency = []
#         for entry in dataset.keys():
#             # if data[entry].idade != "":
#             id.append(entry)
#             idade.append(dataset[entry].idade)
#             gender.append(dataset[entry].gender)
#             text.append(dataset[entry].text)
#             predictedGender.append(dataset[entry].predictedGender)
#             laughFrequency.append(dataset[entry].laughFrequency)
#             emojiFrequency.append(dataset[entry].emojiFrequency)
#             slangFrequency.append(dataset[entry].slangFrequency)
#
#         data = {}
#         data["id"] = id
#         data["idade"] = idade
#         data["gender"] = gender
#         # data["laughFrequency"] = laughFrequency
#         # data["emojiFrequency"] = emojiFrequency
#         # data["slangFrequency"] = slangFrequency
#         data["predictedGender"] = predictedGender
#         data["text"] = text
#         originalDF = pd.DataFrame(data, columns=["id", 'idade', 'gender',
#                                                  "predictedGender", "text"])
#
#         print(datasetsName[i])
#         print(originalDF.shape)
#         originalDF['Characters'] = originalDF.text.str.len()
#         originalDF['Words'] = originalDF.text.str.split().str.len()
#         print("Caracteres")
#         print(originalDF.Characters.sum())
#         print("Palavras")
#         print(originalDF.Words.sum())
#         print("Médias")
#         print(originalDF['Words'].describe())
#         print("Média de caracteres")
#         print(originalDF['Characters'].describe())
#         print("====================================")
#         i = i + 1

if sys.argv[1] == "help":
    print("Primeiro Arg: ")
    print("-------1 - B5-Corpus --------")
    print("-------2 - B2W- Reviews ------")
    print("-------3 - Blogset-BR ------")
    print("-------4 - Stilingue (Deprecated) ------")
    print("-------5 - PAN ------")
    print("-------6 - ESIC ------")
    print("-------7 - BRMORAL ------")
    print("==========================================")
    print("Segundo Arg:")
    print("Predizer apenas gênero: SIM|NAO")
    print("Terceiro Arg:")
    print("Classificador:")
    print("-------1 - B5-Corpus -------")
    print("-------2 - PAN ------")
    print("-------3 - B5-Corpus idade ------")
    print("-------4 - CROSS DOMAIN ------")
    print("Quarto Arg:")
    print("Utilizar Heurística de genero na etapa de treinamento: SIM|NAO")
    print("Quinto Arg:")
    print("Utilizar Heurística de genero na etapa de teste: SIM|NAO")
    quit()

chooseDataSet, predictGenders, chooseClassification,chooseHeuristicaTrain,chooseHeuristicaTest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

heuristica = False
if chooseHeuristicaTrain == "SIM" or chooseHeuristicaTest == "SIM":
    heuristica = True
else:
    heuristica = False
if chooseDataSet == "1":
    data = m.getData(heuristica)
elif chooseDataSet == "2":
    data = m.getReviewData(heuristica)
elif chooseDataSet == "3":
    data = m.getBlogsetData(heuristica)
elif chooseDataSet == "4":
    data = m.getStilingueData(heuristica)
elif chooseDataSet == "5":
    data = m.getAllPan(heuristica)
elif chooseDataSet == "6":
    data = m.getESic(heuristica)
elif chooseDataSet == "7":
    data = m.getBrMoral(heuristica)
else:
    input("Opção inválida, pressione qualquer tecla para finalizar")



id = []
idade = []
gunningIndex = []
gender = []
text = []
predictedGender = []
laughFrequency = []
emojiFrequency = []
slangFrequency = []
for entry in data.keys():
    #if data[entry].idade != "":
    id.append(entry)
    idade.append(data[entry].idade)
    gender.append(data[entry].gender)
    text.append(data[entry].text)
    predictedGender.append(data[entry].predictedGender)
    laughFrequency.append(data[entry].laughFrequency)
    emojiFrequency.append(data[entry].emojiFrequency)
    slangFrequency.append(data[entry].slangFrequency)

data = {}
data["id"] = id
data["idade"] = idade
data["gender"] = gender
#data["laughFrequency"] = laughFrequency
#data["emojiFrequency"] = emojiFrequency
#data["slangFrequency"] = slangFrequency
data["predictedGender"] = predictedGender
data["text"] = text
originalDF = pd.DataFrame(data, columns=["id",'idade', 'gender',
                                         "predictedGender","text"])


#originalDF.to_csv("dataframeGunningIndex.csv",delimiter=";" ,encoding='utf-8', index=True)
#print(originalDF.head())
#ax = originalDF.plot(x="idade", y="gunningIndex")
#ax.figure.savefig('demo-file.pdf')

if predictGenders == "SIM":
    genderManager.printPredictedGenderScores(originalDF)
    quit()

heuristicaTrain = False
heuristicaTest = False
if chooseHeuristicaTrain == "SIM":
    heuristicaTrain = True
if chooseHeuristicaTest == "SIM":
    heuristicaTest = True

if chooseClassification == "1":
    ###### FECHADO!!!!!! #######
    genderManager.b5CorpusPrediction(dataFrame=originalDF,heuristicaTrain=heuristicaTrain ,heuristicaTest=heuristicaTest)
elif chooseClassification == "2":
    ###### FECHADO!!!!!! #######
    genderManager.panCorpusPrediction(originalDF,heuristicaTrain= heuristicaTrain,heuristicaTest=heuristicaTest)
elif chooseClassification == "3":
    originalDF = originalDF.dropna()
    originalDF = originalDF.astype({"idade": int})
    originalDF = originalDF.reset_index(drop=True)
    genderManager.b5CorpusPrediction(dataFrame=originalDF,heuristicaTrain=False,heuristicaTest=False)
elif chooseClassification == "4":
    if chooseDataSet == "3": # Blogset BR
        genderManager.crossDomainPrediction(originalDF,w=600,s="pre",x=600,filter=False,it=200,layers=(300),f="relu",alpha=1e-05,corpusName="blog",corpusThreshold=0.772,genderHeuristicaTrain=heuristicaTrain,genderHeuristicaTest=heuristicaTest)
    elif chooseDataSet == "6": # E-Gov
        genderManager.crossDomainPrediction(originalDF,w=1000,s="pre",x=1000,filter=True,it=200,layers=(500),f="relu",alpha=1e-05,corpusName="e-gov",corpusThreshold=0.780,genderHeuristicaTrain=heuristicaTrain,genderHeuristicaTest=heuristicaTest)
    else: # Opinion
        genderManager.crossDomainPrediction(originalDF,w=100,s="self",x=80,filter=True,it=250,layers=(25,25,25),f="tanh",alpha=1e-07,corpusName="opinion",corpusThreshold=0.745,genderHeuristicaTrain=heuristicaTrain,genderHeuristicaTest=heuristicaTest)


