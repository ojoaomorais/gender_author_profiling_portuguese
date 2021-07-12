from collections import defaultdict
from pathlib import Path

import gensim as gensim
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from tqdm import tqdm
import math

import FeatureExtractManager as featureManager
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score, f1_score
import numpy as np
import nltk
from apyori import apriori
nltk.download('stopwords')
from progress.bar import Bar
from gensim.models import Doc2Vec, Word2Vec
import gensim.downloader as api

def printPredictedGenderScores(dataFrame):
    corretsMasculino = 0
    corretsFeminino = 0
    wrongsMasculino = 0
    wrongsFeminino = 0
    bar = Bar('ChargingBar', max=len(dataFrame.index))

    for index, row in dataFrame.iterrows():
        bar.next()
        predictedGender = featureManager.getGenderStanza(row.text,row.gender)
        if (row.gender is not None) and (predictedGender is not None):
            if row.gender == 0:
                if predictedGender == 1:
                    wrongsMasculino += 1
                elif predictedGender == 0:
                    corretsMasculino += 1
            if row.gender == 1:
                if predictedGender == 1:
                    corretsFeminino += 1
                elif predictedGender == 0:
                    wrongsFeminino += 1
    bar.finish()
    print("NUMERO DE CORRETOS")
    print("Feminino")
    print(corretsFeminino)
    print("Masculino")
    print(corretsMasculino)
    print("NUMERO DE INCORRETOS")
    print("Feminino")
    print(wrongsFeminino)
    print("Masculino")
    print(wrongsMasculino)
    print("Antes do Resample")
    print(dataFrame['idade'].value_counts())
    print(dataFrame['gender'].value_counts())

def getAssociationRules(dataframe):
    rules = []
    for index, row in dataframe.iterrows():
        if row.gender:
            rules.append(featureManager.getAssociationRuleNounsAndGender(row.text,row.gender))
    association_rules = list(apriori(rules, min_support=0.1, min_confidence=0.1, min_lift=0, min_length=2))
    for item in association_rules:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0]
        items = [x for x in pair]
        if len(item) > 2:
            print("Rule: " + items[0] + " -> " + items[1])

            # second index of the inner list
            print("Support: " + str(item[1]))

            # third index of the list located at 0th
            # of the third index of the inner list
            if len(item) > 2:
                print("Confidence: " + str(item[2][0][2]))
            if len(item) > 3:
                print("Lift: " + str(item[2][0][3]))
        else:
            print("Rule: " + items[0])
        print("=====================================")

def b5CorpusPrediction(predictAttribute,dataFrame,genderHeuristica):
    print("Quantidade de amostras: %s", len(dataFrame.text))
    logistic = LogisticRegression(C=25.0,random_state=1)
    final_stopwords_list = stopwords.words('portuguese')
    tfidf = TfidfVectorizer(max_features=3000,stop_words=final_stopwords_list)
    d = tfidf.fit_transform(dataFrame.text)

    #pipeline = Pipeline([('tfidf', tfidf), ('clf', clf)])
    #parameters = [{
    #    "clf__C":[5.0]
    #}]
    #grid_search = GridSearchCV(pipeline, parameters, scoring="f1", cv=10, verbose=10, n_jobs=-1)
    #grid_search.fit(dataFrame.text, dataFrame.gender)
    #print("Best params:")
    #print(grid_search.best_params_)
    #print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
    kFoldClassification(dataFrame,10, logistic,X=d,y=dataFrame.gender,genderHeuristica=genderHeuristica)

def panCorpusPrediction(dataFrame,idadeHeuristica):
    print("Quantidade de amostras: %s",len(dataFrame.text))
    char_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 5),sublinear_tf=True,min_df=2,use_idf=False)
    word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2),sublinear_tf=True,min_df=2,use_idf=False)
    tfidf = FeatureUnion([('char', char_tfidf), ('word', word_tfidf)])
    d = tfidf.fit_transform(dataFrame.text)

    #pipeline = Pipeline([('tfidf', tfidf), ('clf', LinearSVC(C=0.5))])
    #grid_search = GridSearchCV(pipeline,parameters,scoring="accuracy",cv=5,verbose=10,n_jobs=-1)
    #grid_search.fit(dataFrame.text,dataFrame.gender)
    #print("Best params:")
    #print(grid_search.best_params_)
    #print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
    predictedGender = []
    # if (idadeHeuristica):
    #     print("Rodando heurística...")
    #     for index, row in tqdm(dataFrame.iterrows()):
    #         y1 = row.gender
    #         gender = featureManager.getGenderStanza(row.text, y1)
    #         if gender is None:
    #             predictedGender.append(None)
    #         else:
    #             predictedGender.append(gender)
    svc = LinearSVC(C=0.5)
    #if (idadeHeuristica):
    #    data["predictedGender"] = predictedGender
    kFoldClassification(dataFrame,5,svc,X=d,y=dataFrame.gender,genderHeuristica=idadeHeuristica)

from gensim.models import KeyedVectors
preprocess = False

from sklearn.base import TransformerMixin
class TfidfEmbeddingVectorizer(TransformerMixin):
    def __init__(self, model,w):
        self.model = model
        self.w = w

    def fit(self, X, y=None, **fit_params):
        tfidf = TfidfVectorizer()
        xConverted = X.astype('U').values
        tfidf.fit(xConverted)
        self.max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(int, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X, **fit_params):
        out_m = []
        i = 0
        for text in tqdm(X):
            i = i + 1
            parag_M = []
            #Com Word Embeddings do Google
            for token in gensim.utils.simple_preprocess(text):
            #Com Word embeddings Construídos
            #for token in text.split():
                if token in self.model:
                    if token in self.word2weight:
                        parag_M.append(self.model[token] * self.word2weight[token])
                    else:
                        parag_M.append(self.model[token] * self.max_idf)
            if parag_M:
                out_m.append(np.average(parag_M, axis=0))
            else:
                out_m.append([0]*self.w)
        return np.array(out_m)

    def fit_transform(self, X,**fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    for i, line in enumerate(input_file):

        if (i % 10000 == 0):
            print("read {0} reviews".format(i))
        # do some pre-processing and return a list of words for each review text
        yield gensim.utils.simple_preprocess(line)

def crossDomainPrediction(dataFrame,w,s,x,filter,it,layers,f,alpha,corpusName,corpusThreshold,genderHeuristica):
        model = None
        X_train = None
        X_test = None
        y_train = None
        y_test = None
        xtrainName = "XTrain" + corpusName + ".csv"
        xtestName = "XTest" + corpusName + ".csv"
        ytrainName = "YTrain" + corpusName + ".csv"
        ytestName = "YTest" + corpusName + ".csv"
        if (s == "pre"):
            if (w == 100):
                model = KeyedVectors.load_word2vec_format("word2vec100.txt")
            elif(w == 600):
                model = KeyedVectors.load_word2vec_format("word2vec600.txt")
            else:
                model = KeyedVectors.load_word2vec_format("word2vec1000.txt")
        else:
            if (Path(xtrainName).is_file() == False) and (Path(xtestName).is_file() == False) and (Path(ytrainName).is_file() == False) and (Path(ytestName).is_file() == False):
                documents = list(read_input(dataFrame.text))
                model = gensim.models.Word2Vec(documents, vector_size=100)
                model.save('word2vecOpinion.kv')
            model = Word2Vec.load('word2vecOpinion.kv').wv
        mlp = MLPClassifier(solver="lbfgs", hidden_layer_sizes=layers, random_state=1, max_iter=it, activation=f,alpha=alpha)
        if Path(xtrainName).is_file() and Path(xtestName).is_file() and Path(ytrainName).is_file() and Path(ytestName).is_file() :
            np.random.seed(1)
            print("Existe")
            X_train = pd.read_csv(xtrainName, header=None, squeeze=True, index_col=0)
            X_test = pd.read_csv(xtestName, header=None, squeeze=True, index_col=0)
            y_train = pd.read_csv(ytrainName, header=None, squeeze=True, index_col=0)
            y_test = pd.read_csv(ytestName, header=None, squeeze=True, index_col=0)
            X_train = X_train.iloc[1:]
            X_test = X_test.iloc[1:]
            y_train = y_train.iloc[1:]
            y_test = y_test.iloc[1:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(dataFrame.text, dataFrame.gender, test_size=0.2,
                                                                stratify=dataFrame.gender)
        tfidfVec = TfidfEmbeddingVectorizer(model=model, w=w)
        X_train = X_train.fillna(' ')
        X_test = X_test.fillna(' ')
        X_train_vec = tfidfVec.fit_transform(X_train)
        mlp.fit(X_train_vec, y_train)
        X_test_vec = tfidfVec.transform(X_test)
        predict = mlp.predict(X_test_vec)
        f1 = f1_score(y_test, predict, average='micro')
        print("#### PRE HEURISTICA ####")
        print('fscore:    {}'.format(f1))
        print("########################")

        if genderHeuristica:
            k = 0
            for index, row in X_test.iteritems():
                ind = int(index)
                gender = dataFrame.iloc[ind].predictedGender
                if not math.isnan(gender):
                    predict[k] = gender
                k = k + 1
            f1 = f1_score(y_test, predict, average='micro')
            print("#### PÓS HEURISTICA ####")
            print('fscore:    {}'.format(f1))
            print("########################")


def kFoldClassification(df,foldNumber,classificator,X,y,genderHeuristica):

    kf = StratifiedKFold(n_splits=foldNumber, shuffle=True, random_state=1)
    totalPrecision = []
    totalF1 = []
    totalPrecisionHeuristic = []
    totalF1Heuristic = []
    i = 1
    for train_index, test_index in kf.split(X, y):
        print('{} of KFold {}'.format(i, kf.n_splits))
        dfTest = df.loc[test_index]
        xtr, xvl = X[train_index], X[test_index]
        ytr, yvl = y[train_index], y[test_index]
        # if (genderHeuristica):
        #     for index, row in dfTrain.iterrows():
        #         ind = dfTrain[dfTrain.text == row.text].index[0]
        #         y1 = row.gender
        #         if (math.isnan(row.predictedGender) == False) and (row.predictedGender != y1):
        #             print("================ ROW ===================")
        #             print(row)
        #             print("=================YTR====================")
        #             print(ytr[ind])
        #             print("=================INDEX==================")
        #             print(ind)
        #             print("================XTR=====================")
        #             print(xtr)
        #             xtr = xtr.drop(ind)
        #             ytr = ytr.drop(ind)
        classificator.fit(xtr, ytr)
        predicts = classificator.predict(xvl)
        k = 0
        precision, recall, fscore, support = score(yvl, predicts)
        totalPrecision.append(precision)
        totalF1.append(fscore)
        print("#### PRE HEURISTICA ####")
        print('precision: {}'.format(precision))
        print('fscore:    {}'.format(fscore))
        print("########################")
        if(genderHeuristica):
            k = 0
            for index, row in dfTest.iterrows():
                gender = row.predictedGender
                if not math.isnan(gender):
                    predicts[k] = gender
                k = k + 1
            precision, recall, fscore, support = score(yvl, predicts)
            totalPrecisionHeuristic.append(precision)
            totalF1Heuristic.append(fscore)
            print("#### POS HEURISTICA ####")
            print('precision: {}'.format(precision))
            print('fscore:    {}'.format(fscore))
            print("########################")
        i += 1
    print("#### RESULTADO FINAL PRE HEURISTICA####")
    axisCount = len(totalPrecision[0])
    for axis in range(axisCount):
        valuesArray = []
        for value in totalPrecision:
            valuesArray.append(value[axis])
        print("Precisão Classe: ",axis)
        print(np.mean(valuesArray))
        valuesArray = []
        for value in totalF1:
            valuesArray.append(value[axis])
        print("Fscore Class: ",axis)
        print(np.mean(valuesArray))
    if(genderHeuristica):
        print("#######################################")
        print("#### RESULTADO FINAL POS HEURISTICA####")
        axisCount = len(totalPrecisionHeuristic[0])
        for axis in range(axisCount):
            valuesArray = []
            for value in totalPrecisionHeuristic:
                valuesArray.append(value[axis])
            print("Precisão Classe: ", axis)
            print(np.mean(valuesArray))
            valuesArray = []
            for value in totalF1Heuristic:
                valuesArray.append(value[axis])
            print("Fscore Class: ", axis)
            print(np.mean(valuesArray))
        print("#######################################")