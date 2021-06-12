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

# def getGender(text,realGender):
#     nlp = spacy.load("pt_core_news_sm")
#     femCount = 0
#     maleCount = 0
#     doc = nlp(text)
#     sub_toks = [tok for tok in doc if ("Person=1" in tok.tag_ and "Number=Sing" in tok.tag_)]
#     if len(sub_toks) > 0:
#         for token in sub_toks:
#             print("Token.Lemma in LinkingVerbs")
#             print(token.lemma_ in linkingVerbs)
#             print("AUX")
#             print("AUX AUX__Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin" in token.tag_)
#             print(token)
#             if (token.lemma_ in linkingVerbs) or ("AUX AUX__Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin" in token.tag_):
#                 gend = updateGenderCount(doc,token,token.head)
#                 if gend:
#                     if gend == "Masc":
#                         maleCount += 1
#                     elif gend == "Fem":
#                         femCount += 1
#                 for child in token.children:
#                     gend = updateGenderCount(doc, token, child)
#                     if gend:
#                         if gend == "Masc":
#                             maleCount += 1
#                         elif gend == "Fem":
#                             femCount += 1
#         if maleCount == 0 and femCount == 0 or maleCount == femCount:
#             return
#         if maleCount > femCount:
#             return 0
#         else:
#             return 1
#     return

stanza.download('pt')
nlpStanza = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos,lemma,depparse')

def getGenderStanza(text,realGender):
    femCount = 0
    maleCount = 0
    sentences = sent_tokenize(text)
    for s in sentences:
        doc = nlpStanza(s)
        for d in doc.sentences:
            d.text = d.text.strip()
            if (d.text.startswith("\" ") or d.text.endswith(" \"")) == False:
                for word in d.words:
                    if word.feats:
                        if ("Person=1" in word.feats and "Number=Sing" in word.feats):
                            if (word.lemma in linkingVerbs):
                                if word.head > 0:
                                    if d.words[word.head - 1].feats:
                                        if not isBeforeOrDistant(d.text, word.text, d.words[word.head-1].text):
                                            gend = generoStanza(feats=d.words[word.head - 1].feats,pos=d.words[word.head - 1].pos,
                                                                verb=True,adj=True,noun=True,word=d.words[word.head-1].text,lemma=d.words[word.head-1].lemma)
                                            if gend:
                                                if gend == "Masc":
                                                    maleCount += 1
                                                elif gend == "Fem":
                                                    femCount += 1
                                for child in d.words:
                                    if not isBeforeOrDistant(d.text, word.text, child.text):
                                        if child.head == word.id:
                                            if child.feats:
                                                gend = generoStanza(feats=child.feats, pos=child.pos,
                                                                    verb=True, adj=True, noun=True,word=child.text,lemma=child.lemma)
                                                if gend:
                                                    if gend:
                                                        if gend == "Masc":
                                                            maleCount += 1
                                                        elif gend == "Fem":
                                                            femCount += 1
    if maleCount == 0 and femCount == 0 or maleCount == femCount:
        return None
    treshold = 0.75
    if maleCount > femCount:
        if (maleCount/(maleCount + femCount)) > treshold:
            return 0
    else:
        if(femCount/(maleCount + femCount)) > treshold:
            return 1

def getAssociationRuleNounsAndGender(text,gender):
    nouns = []
    nlp = spacy.load("pt_core_news_sm")
    doc = nlp(text)
    for possible_noun in doc:
        if(possible_noun.pos_ == "NOUN"):
            nouns.append(possible_noun.text)
    nouns.append(str(gender))
    return list(nouns)



def updateGenderCount(doc,token,related):
    if not isBeforeOrDistant(doc, token,related):
        isCheckNoun = (token.lemma_ == "ser")
        gender = genero(related, adj=True, verb=True, noun=isCheckNoun)
        if gender:
            return gender
    return

def investigateGenderPossibilities(dataFrame):
    nlp = spacy.load("pt_core_news_sm")
    stanza.download('pt')
    for entry in list(dataFrame.text):
        print("==============================")
        #doc = nlp("recebi o produto estou insastifeita e muito chateada")
        gend = None
        nlp = stanza.Pipeline(lang='pt', processors='tokenize,mwt,pos,lemma,depparse')
        doc = nlp("recebi o produto estou insastifeita e muito chateada")
        print(doc)


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
    #if noun:
    #    g = getNounStanza(feats=feats,pos=pos,word=word)
    #    if g: return g
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

##############################################################################################################################################################

################################ FEATURES IDADE ##############################################################################################################
import string
def getTokenized(text):
    cleanedText = re.sub("\$.*?\$", '', text)
    table = str.maketrans(dict.fromkeys(string.punctuation))
    cleanedText = cleanedText.translate(table)
    tokens = nltk.tokenize.word_tokenize(cleanedText)
    return tokens

import emojis
emojiKeyboardList = [":‑)", ":)", ":-]", ":]", ":-3", ":3", ":->", ":>", "8-)", "8)", ":-}", ":}", ":o)", ":c)", ":^)", "=]", "=)", ":‑D", ":D", "8‑D", "8D", "x‑D", "xD", "X‑D", "XD", "=D", "=3", "B^D", ":-))", ":‑(", ":(", ":‑c", ":c", ":‑<", ":<", ":‑[", ":[", ":-||", ">:[", ":{", ":@", ":(", ";(", ":'‑(", ":'(", ":'‑)", ":')", "D‑':", "D:<", "D:", "D8", "D;", "D=", "DX", ":‑O", ":O", ":‑o", ":o", ":-0", "8‑0", ">:O", ":-*", ":*", ":×", ";‑)", ";)", "*-)", "*)", ";‑]", ";]", ";^)", ":‑,", ";D", ":‑P", ":P", "X‑P", "XP", "x‑p", "xp", ":‑p", ":p", ":‑Þ", ":Þ", ":‑þ", ":þ", ":‑b", ":b", "d:", "=p", ">:P", ":‑/", ":/", ":‑.", ">:\\", ">:/", ":\\", "=/", "=\\", ":L", "=L", ":S", ":‑|", ":|", ":$", ":/", "/)", "://3", ":‑X", ":X", ":‑#", ":#", ":‑&", ":&", "O:‑)", "O:)", "0:‑3", "0:3", "0:‑)", "0:)", "0;^)", ">:‑)", ">:)", "}:‑)", "}:)", "3:‑)", "3:)", ">;)", ">:3", ";3", "|;‑)", "|‑O", ":‑J", "#‑)", "%‑)", "%)", ":‑###..", ":###..", "<:‑|", "',:-|", "',:-l", ":E"]
def getEmojiFrequency(text):
    count = 0
    splittedText = text.split()
    for word in splittedText:
        if word in emojiKeyboardList:
            count += 1
    count += emojis.count(text)
    return count / len(splittedText)

laughs = ["k{3}","(?:ha){2}","(?:he){2}","(?:huas){2}","(?:hue){2}","(?:rs){1}"]
import re
def getLaughFrequency(text):
    frequency = 0
    splittedText = getTokenized(text)
    for l in laughs:
        regex = re.findall(l, " ".join(splittedText))
        frequency += len(regex) / len(splittedText)
    return frequency

slangs= ["vcs","oq","pprt","tmj","pdp","slc","slk","mb","mvd","mv","sv","sfd","plmns","plmd","flw","rd","vdb","smdd","tqr","tlg","dmr","pdc","sqn","lol","omg","pls","tks",'wtf',"nsfw","pdsm","tbt","bff","diy","f4f","pvt","sdv","fb","wpp","tt","ig","smp","agr","amg","bb","tds","blz","cvs","ngm","sdds","dlç","obg","glr","msm","td","tdb","bjs" ]
nsfwWords = ['Anus','Baba-ovo','Babaovo','Babaca','Bacura','Bagos','Baitola','Bebum','Besta','Bicha','Bisca','Bixa','Boazuda','Boceta','Boco','Boiola','Bolagato','Boquete','Bolcat','Bosseta','Bosta','Bostana','Brecha','Brexa','Brioco','Bronha','Buca','Buceta','Bunda','Bunduda','Burra','Burro','Busseta','Cachorra','Cachorro','Cadela','Caga','Cagado','Cagao','Cagona','Canalha','Caralho','Casseta','Cassete','Checheca','Chereca','Chibumba','Chibumbo','Chifruda','Chifrudo','Chota','Chochota','Chupada','Chupado','Clitoris','Cocaina','Coco','Corna','Corno','Cornuda','Cornudo','Corrupta','Corrupto','Cretina','Cretino','Cruz-credo','Cu','Culhao','Curalho','Cuzao','Cuzuda','Cuzudo','Debil','Debiloide','Defunto','Demonio','Difunto','Doida','Doido','Egua','Escrota','Escroto','Esporrada','Esporrado','Esporro','Estupida','Estupidez','Estupido','Fedida','Fedido','Fedor','Fedorenta','Feia','Feio','Feiosa','Feioso','Feioza','Feiozo','Felacao','Fenda','Foda','Fodao','Fode','FodidaFodido','Fornica','Fudendo','Fudecao','Fudida','Fudido','Furada','Furado','Furão','Furnica','Furnicar','Furo','Furona','Gaiata','Gaiato','Gay','Gonorrea','Gonorreia','Gosma','Gosmenta','Gosmento','Grelinho','Grelo','Homo-sexual','Homossexual','Homossexual','Idiota','Idiotice','Imbecil','Iscrota','Iscroto','Japa','Ladra','Ladrao','Ladroeira','Ladrona','Lalau','Leprosa','Leproso','Lésbica','Macaca','Macaco','Machona','Machorra','Manguaca','Mangua¦a','Masturba','Meleca','Merda','Mija','Mijada','Mijado','Mijo','Mocrea','Mocreia','Moleca','Moleque','Mondronga','Mondrongo','Naba','Nadega','Nojeira','Nojenta','Nojento','Nojo','Olhota','Otaria','Ot-ria','Otario','Ot-rio','Paca','Paspalha','Paspalhao','Paspalho','Pau','Peia','Peido','Pemba','Pênis','Pentelha','Pentelho','Perereca','Peru','Pica','Picao','Pilantra','Piranha','Piroca','Piroco','Piru','Porra','Prega','Prostibulo','Prost-bulo','Prostituta','Prostituto','Punheta','Punhetao','Pus','Pustula','Puta','Puto','Puxa-saco','Puxasaco','Rabao','Rabo','Rabuda','Rabudao','Rabudo','Rabudona','Racha','Rachada','Rachadao','Rachadinha','Rachadinho','Rachado','Ramela','Remela','Retardada','Retardado','Ridícula','Rola','Rolinha','Rosca','Sacana','Safada','Safado','Sapatao','Sifilis','Siririca','Tarada','Tarado','Testuda','Tezao','Tezuda','Tezudo','Trocha','Trolha','Troucha','Trouxa','Troxa','Vaca','Vagabunda','Vagabundo','Vagina','Veada','Veadao','Veado','Viada','Viado','Viadao','Xavasca','Xerereca','Xexeca','Xibiu','Xibumba','Xota','Xochota','Xoxota','Xana','Xaninha']
def getSlangFrequency(text):
    count = 0
    splittedText = getTokenized(text)
    for word in splittedText:
    	if (word.lower() in slangs) or (word.capitalize() in nsfwWords):
            count += 1
    return count / len(splittedText)

def getAge(text):
    index = gunningFogIndex(text)
    return index

# def gunningFogIndex(text):
#     number_of_sentences = sent_tokenize(text)
#     sentenceCount = len(number_of_sentences)
#     regex = re.compile('[^A-Za-zéêíîôóõáàâãúûçÉÊÍÎÔÓÕÁÀÂÃÚÛÇ]+')
#     cleanedText = re.sub(regex, ' ', text)
#     wordCount = len(cleanedText.split())
#     complexWords = 0
#     for word in cleanedText.split():
#         sCount = syllableCount(word)
#         if sCount >= 3:
#             complexWords = complexWords + 1
#     index = 0
#     if(sentenceCount > 0 and wordCount > 0):
#         index = 0.4 * ((wordCount/sentenceCount) + (100 * (complexWords/wordCount)))
#     return float("{:.2f}".format(index))

#import spacy
#from spacy_syllables import SpacySyllables

#nlp = spacy.load("pt_core_news_sm")
#syllables = SpacySyllables(nlp)
isSyllableFirstTime = True

#def syllableCount(word):
#    if "syllables" in nlp.pipe_names:
#        nlp.remove_pipe("syllables")
    #nlp.add_pipe(syllables, after="tagger")
    #doc = nlp(word)
    #data = [(token.text, token._.syllables, token._.syllables_count) for token in doc]
    #return data[0][2]


##############################################################################################################################################################

############################### FEATURES GENERO ##############################################################################################################
