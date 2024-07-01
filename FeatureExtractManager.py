import nltk as nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('genesis')
nltk.download('universal_tagset')
nltk.download('wordnet')
nltk.download('omw')
import stanza

################################ HEURISTICA DE GENERO ########################################################################################################

adjetivosUniformes = ['e', 'l', 'm', 'r', 's','z']
linkingVerbs = ["ser","estar","permanecer","ficar","parecer","andar","viver","virar","continuar"]
adjBifFormaNormal1 = ["o","u","ês","or","ão","éu","eu"]
adjBifFormaFem = ["a","ã","ona","oa","eia"]
subBifMasc = ["homem","pai","avô","menino","aluno","autor"]
subBifFem = ["mulher","mãe","avó","menina","aluna","autora"]
adjPej = ["contigo","convosco","consumista","vegetariano","ciclista","alvo","criança","dentista","humano","petralha","petista","foda","canalha","bicha","idiota","babaca","patriota","hetero","hétero","gay"]

stanza.download('pt')
nlpStanza = stanza.Pipeline(lang='pt', 
                            processors='tokenize,mwt,pos,lemma,depparse')

def getGenderStanza(text,realGender):
    femCount = 0
    maleCount = 0
    sentences = sent_tokenize(text)
    for s in sentences:
        doc = nlpStanza(s)
        for d in doc.sentences:
            d.text = d.text.strip()
            if (d.text.startswith("\" ") or 
                d.text.endswith(" \"")) == False:
                for word in d.words:
                    if word.feats:
                        if ("Person=1" in word.feats 
                            and "Number=Sing" in word.feats):
                            if (word.lemma in linkingVerbs):
                                if word.head > 0:
                                    if d.words[word.head - 1].feats:
                                        if not isBeforeOrDistant(d.text, 
                                                                 word.text, 
                                                                 d.words[word.head-1].text):
                                            gend = generoStanza(feats=d.words[word.head - 1].feats,
                                                                pos=d.words[word.head - 1].pos,
                                                                verb=True,adj=True,noun=True,
                                                                word=d.words[word.head-1].text,
                                                                lemma=d.words[word.head-1].lemma)
                                            if gend:
                                                if gend == "Masc":
                                                    maleCount += 1
                                                elif gend == "Fem":
                                                    femCount += 1
                                for child in d.words:
                                    if not isBeforeOrDistant(d.text, 
                                                             word.text, 
                                                             child.text):
                                        if child.head == word.id:
                                            if child.feats:
                                                gend = generoStanza(feats=child.feats, 
                                                                    pos=child.pos,
                                                                    verb=True, 
                                                                    adj=True, 
                                                                    noun=True,
                                                                    word=child.text,lemma=child.lemma)
                                                if gend:
                                                    if gend:
                                                        if gend == "Masc":
                                                            maleCount += 1
                                                        elif gend == "Fem":
                                                            femCount += 1
    if maleCount == 0 and femCount == 0 or maleCount == femCount:
        return
    treshold = 0.75
    if maleCount > femCount:
        if (maleCount/(maleCount + femCount)) > treshold:
            return 0
    else:
        if(femCount/(maleCount + femCount)) > treshold:
            return 1

def updateGenderCount(doc,token,related):
    if not isBeforeOrDistant(doc, token,related):
        isCheckNoun = (token.lemma_ == "ser")
        gender = genero(related, adj=True, verb=True, noun=isCheckNoun)
        if gender:
            return gender
    return

def getPhrase(doc,token,related):
    k = 0
    j = 0
    indexToken = 0
    indexRelated = 0
    for t in doc:
        if t == token:
            indexToken = k
        if t == related:
            indexRelated = j
        k = k + 1
        j = j + 1
    print("Is Before: ",(indexRelated < indexToken))
    if indexRelated == indexToken:
        return ""
    if(indexRelated > indexToken):
        return doc[indexToken:indexRelated + 5]

def isBeforeOrDistant(doc,token,related):
    k = 0
    j = 0
    indexToken = 0
    indexRelated = 0
    spplited = doc.split()
    for t in spplited:
        if t == token:
            indexToken = k
        if t == related:
            indexRelated = j
        k = k + 1
        j = j + 1
    return indexRelated < indexToken or (indexRelated - indexToken > 3)

def genero(pos,adj,verb,noun):
    g = None
    if adj:
        g = getAdj(pos)
        if g: return g
    if verb:
        g = getVerb(pos)
        if g: return g
    if noun:
        g = getNoun(pos)
        if g: return g
    return g

def generoStanza(word,lemma,feats,pos,adj,verb,noun):
    g = None
    if word in string.punctuation:
        return g
    if adj:
        g = getAdjStanza(feats=feats,pos=pos,word=word,lemma=lemma)
        if g: return g
    if verb:
        g = getVerbStanza(feats=feats,pos=pos,lemma=lemma)
        if g: return g
    return g

def getNoun(pos):
    search = re.search('NOUN__Gender=(.+?)|Number=Sing', pos.tag_)
    if search:
        if pos.text.lower() in subBifMasc:
            return "Masc"
        if pos.text.lower() in subBifFem:
            return "Fem"
    return

def getAdj(pos):
    search = re.search('ADJ__Gender=(.+?)\|Number=Sing', pos.tag_)
    if search:
        if pos.text[-1:] in adjetivosUniformes \
                         or pos.text.endswith("ndo") \
                         or pos.text.endswith("nda") \
                         or pos.lemma_.lower() in adjPej\
                         or pos.text.lower() in adjPej:
            return
        if search.group(1) == "Fem":
            # VERIFICACAO EM DUAS ETAPAS
            for adjBifFem in adjBifFormaFem:
                if pos.text[-len(adjBifFem):] in adjBifFormaFem:
                    return "Fem"
            return
        elif search.group(1) == "Masc":
            for adjBifMasc in adjBifFormaNormal1:
                if pos.text[-len(adjBifMasc):] in adjBifFormaNormal1:
                    return "Masc"
            return
        else:
            return

def getVerb(pos):
    search = re.search('VERB__Gender=(.+?)\|Number=Sing\|VerbForm=Part', pos.tag_)
    if search:
        if pos.lemma_ in adjPej:
            return
        if search.group(1) == "Fem":
            return "Fem"
        elif search.group(1) == "Masc":
            return "Masc"
        else:
            return

def getNounStanza(word,pos,feats):
    search = re.search('Gender=(.+?)|Number=Sing', feats)
    if search and pos == "NOUN":
        if word.lower() in subBifMasc:
            return "Masc"
        if word.lower() in subBifFem:
            return "Fem"
    return

def getAdjStanza(word,lemma,pos,feats):
    search = re.search('Gender=(.+?)\|Number=Sing', feats)
    if search and pos == "ADJ":
        if word[-1:] in adjetivosUniformes \
                         or word.endswith("ndo") \
                         or word.endswith("nda") \
                         or lemma.lower() in adjPej\
                         or word.lower() in adjPej:
            return
        if search.group(1) == "Fem":
            # VERIFICACAO EM DUAS ETAPAS
            for adjBifFem in adjBifFormaFem:
                if word[-len(adjBifFem):] in adjBifFormaFem:
                    return "Fem"
            return
        elif search.group(1) == "Masc":
            for adjBifMasc in adjBifFormaNormal1:
                if word[-len(adjBifMasc):] in adjBifFormaNormal1:
                    return "Masc"
            return
        else:
            return

def getVerbStanza(lemma,pos,feats):
    search = re.search('Gender=(.+?)\|Number=Sing\|VerbForm=Part', feats)
    if search and pos == "VERB":
        if lemma in adjPej:
            return
        if search.group(1) == "Fem":
            return "Fem"
        elif search.group(1) == "Masc":
            return "Masc"
        else:
            return
    return

############################### FEATURES GENERO ##############################################################################################################
