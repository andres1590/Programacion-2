#Importación modulos

import pandas as pd
import numpy as np
import re
import nltk
from nltk import ngrams
import spacy
import mlflow

df_glass_completo=pd.read_csv("/Users/andresfelipe/Documents/Prog2/glassdoor_reviews.csv")

#Se hace un DF mas pequeño para trabajar mas facilemte
df_glass_mil =df_glass_completo.head(1000)

df_glass_mil.to_csv('/Users/andresfelipe/Documents/Programacion-2/glassdor_mil.csv', index=False)

#Crea df con prueba y test
train_set = df_glass_mil.sample(frac=0.8, random_state=42) 
train_set.to_csv('/Users/andresfelipe/Documents/Programacion-2/glassdor_train.csv', index=False)

# Drop de los datos que estan en el set train
 
test_set = df_glass_mil.drop(train_set.index) 
test_set.to_csv('/Users/andresfelipe/Documents/Programacion-2/glassdor_train.csv', index=False)

#Se trabaja con el train df

df_glass=pd.read_csv("/Users/andresfelipe/Documents/Programacion-2/glassdor_train.csv")

#Replace NaN with , Na values interfere in the merge and re code

df_glass['headline'] = df_glass['headline'].fillna(",")
df_glass['pros'] = df_glass['pros'].fillna(",")
df_glass['cons'] = df_glass['cons'].fillna(",")

#Merge hedline, pros, cons
df_glass["headline+pros+cons"] = df_glass["headline"] + " " + df_glass["pros"]+ " " + df_glass["cons"]

#Make the info at column "headline+pros+cons" string
df_glass['headline+pros+cons']= df_glass['headline+pros+cons'].astype('string')

#Remove Punctuation Marks

def quitar_puntuacion(texto):
    return re.sub(r'[^\w\s]',"", texto)

# Apply function to the merged column

df_glass['headline+pros+cons'] = df_glass['headline+pros+cons'].apply(quitar_puntuacion)

#Remove Stop words
def quitar_stop(texto):
    return re.sub(r'\b(?:and|an|at|a|of|on|I|for|with|the|at|from|in|to)\b', "", texto, flags=re.IGNORECASE)

# Apply function to the merged column

df_glass['headline+pros+cons'] = df_glass['headline+pros+cons'].apply(quitar_stop)

#Lematize
nlp = spacy.load("en_core_web_sm")

def lematizar_(text):
    obj = nlp(text.lower())
    lemmatize_token = [x.lemma_ for x in obj]
    return lemmatize_token

# Aplicamos la lematización a cada fila
df_glass['headline+pros+cons_lemma'] = df_glass['headline+pros+cons'].apply(lematizar_)

#Hacer lemma string
df_glass['headline+pros+cons_lemma']=df_glass['headline+pros+cons_lemma'].astype('string')

#Remove Symbols Stop words
def quitar_stop(texto):
    return re.sub(r"[ \[\],']", " ", texto, flags=re.IGNORECASE)

# Apply function to the merged column

df_glass['headline+pros+cons_cleanlemma'] = df_glass['headline+pros+cons_lemma'].apply(quitar_stop)

#Create new DF with bigrams to record Y values (recommend)

df_bigrams_recommend=pd.DataFrame({"count":[],"Recommend":[]})

cc = 0

for x in df_glass['headline+pros+cons_cleanlemma']:

    text = df_glass['headline+pros+cons_cleanlemma'].iloc[cc]
    #recommend = df_glass['recommend'].iloc[cc]

    tokens = nltk.word_tokenize(text)

    unigrams=list(ngrams(tokens, 1))
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    Ngrams=-1 #To record all

    #unigram_counts = (pd.Series(unigrams).value_counts())[:Ngrams]

    bigram_counts = (pd.Series(bigrams).value_counts())[:Ngrams]
    
    #trigram_counts = (pd.Series(trigrams).value_counts())[:Ngrams]
    

    df_new = bigram_counts.to_frame()

    df_new["Recommend"]= df_glass['recommend'].iloc[cc]

    df_bigrams_recommend = pd.concat([df_new, df_bigrams_recommend])
    cc = cc+ 1
#Index a columna
df_bigrams_recommend=df_bigrams_recommend.reset_index()
#Save the DF with bigrams on a CSV
df_bigrams_recommend.to_csv('Bigrams.csv', index=False)
print("Bigramas Train")
print(df_bigrams_recommend)
#Se quitan los duplicados del reporte de recommend, y se remueve la columna count

valores_bigrams_unicos = df_bigrams_recommend.drop_duplicates(subset=['index','Recommend'])
valores_bigrams_unicos = valores_bigrams_unicos.drop('count',axis=1)
#Convertimos los valores de la tupla a string, y depues concatenamos

valores_bigrams_unicos["index"]=valores_bigrams_unicos["index"].astype(str)
valores_bigrams_unicos["index+Recommend"]=valores_bigrams_unicos["index"]+ valores_bigrams_unicos["Recommend"]

#Agarramos el df de las recomendaciones y hacemos el concatenado anterior

df_bigrams_recommend_modificado=df_bigrams_recommend
df_bigrams_recommend_modificado["index"]=df_bigrams_recommend_modificado["index"].astype(str)
df_bigrams_recommend_modificado["index+Recommend"]=df_bigrams_recommend_modificado["index"]+ df_bigrams_recommend_modificado["Recommend"]

#Hacemos un ciclo For para hacer la suma acumulada "Frecuencia Bigrama + Reseña"

valores_bigrams_unicos["Frecuencia Bigrama + Reseña"]=0

colcount= df_bigrams_recommend_modificado.columns.get_loc('count')
colBR=  valores_bigrams_unicos.columns.get_loc("Frecuencia Bigrama + Reseña")


loc=0

for gramm in valores_bigrams_unicos['index+Recommend']:
    cc=0
    loc2=0
    
    for gramm_2 in df_bigrams_recommend_modificado ['index+Recommend']:
        if gramm == gramm_2:
            cc= cc+ df_bigrams_recommend_modificado.iloc[loc2,colcount]
        loc2=loc2+1
    
    valores_bigrams_unicos.iloc[loc,colBR]=cc
    loc=loc+1
#Hacemos un ciclo For para hacer la suma acumulada "Frecuencia Bigrama"
valores_bigrams_unicos["Frecuencia Bigrama"]=0

colcount= df_bigrams_recommend_modificado.columns.get_loc('count')
fB= valores_bigrams_unicos.columns.get_loc("Frecuencia Bigrama")

loc=0

for gramm in valores_bigrams_unicos['index']:
    cc=0
    loc2=0
    
    for gramm_2 in df_bigrams_recommend_modificado ['index']:
        if gramm == gramm_2:
            cc= cc+ df_bigrams_recommend_modificado.iloc[loc2,colcount]
        loc2=loc2+1
    
    valores_bigrams_unicos.iloc[loc,fB]=cc
    loc=loc+1
#Hacemos un ciclo For para calcular probabilidad

valores_bigrams_unicos["Probabilidad"]= None
loc=0
colprob= valores_bigrams_unicos.columns.get_loc('Probabilidad')
colFBR= valores_bigrams_unicos.columns.get_loc("Frecuencia Bigrama + Reseña")
colFB= valores_bigrams_unicos.columns.get_loc('Frecuencia Bigrama')

for fecuencia in valores_bigrams_unicos['index']:

    valores_bigrams_unicos.iloc[loc,colprob]=valores_bigrams_unicos.iloc[loc,colFBR]/valores_bigrams_unicos.iloc[loc,colFB]
    loc=loc+1
valores_bigrams_unicos.to_csv('ProbabilidadBiGramm.csv', index=False)

#Hacer tabla prob limpia
df_bigrams_test=pd.DataFrame({"Word":[],"V":[],"X":[],"O":[]})
df_bigrams_test['Word']=valores_bigrams_unicos['index'].drop_duplicates()
df_bigrams_test = df_bigrams_test.fillna(0)

colP=valores_bigrams_unicos.columns.get_loc("Probabilidad")
colRec=valores_bigrams_unicos.columns.get_loc("Recommend")
colV=df_bigrams_test.columns.get_loc("V")
colX=df_bigrams_test.columns.get_loc("X")
colO=df_bigrams_test.columns.get_loc("O")



locw=0
for word in df_bigrams_test['Word']:
    loci=0
    for index in valores_bigrams_unicos['index']:
        if word == index:
            if valores_bigrams_unicos.iloc[loci,colRec]=='v':
                df_bigrams_test.iloc[locw,colV]=valores_bigrams_unicos.iloc[loci,colP]
            if valores_bigrams_unicos.iloc[loci,colRec]=='x':
                df_bigrams_test.iloc[locw,colX]=valores_bigrams_unicos.iloc[loci,colP]
            if valores_bigrams_unicos.iloc[loci,colRec]=='o':
                df_bigrams_test.iloc[locw,colO]=valores_bigrams_unicos.iloc[loci,colP]
        loci=loci+1
    locw=locw+1 
df_bigrams_test.to_csv('ProbabilidadBiGramm_VXO.csv', index=False)

print("Listado de Probabilidades Bigram Train")
print(df_bigrams_test)

#Create new DF with unigrams to record Y values (recommend)

colhpccl= df_glass.columns.get_loc('headline+pros+cons_cleanlemma')
colrm= df_glass.columns.get_loc('recommend')


df_uigrams_recommend=pd.DataFrame({"count":[],"Recommend":[]})

cc = 0

for x in df_glass['headline+pros+cons_cleanlemma']:

    text = df_glass.iloc[cc,colhpccl]
    #recommend = df_glass['recommend'].iloc[cc]

    tokens = nltk.word_tokenize(text)

    unigrams=list(ngrams(tokens, 1))
    #bigrams = list(ngrams(tokens, 2))
    #trigrams = list(ngrams(tokens, 3))

    Ngrams=-1 #To record all

    unigram_counts = (pd.Series(unigrams).value_counts())[:Ngrams]

    #bigram_counts = (pd.Series(bigrams).value_counts())[:Ngrams]
    
    #trigram_counts = (pd.Series(trigrams).value_counts())[:Ngrams]
    

    df_new = unigram_counts.to_frame()

    df_new["Recommend"]= df_glass.iloc[cc,colrm]

    df_uigrams_recommend = pd.concat([df_new, df_uigrams_recommend])
    cc = cc+ 1
#Index a columna
df_uigrams_recommend = df_uigrams_recommend.reset_index()

#Save the DF with unigrams on a CSV
df_uigrams_recommend.to_csv('Unigrams.csv', index=False)
print("Unigramas Train")
print(df_uigrams_recommend)

#Se quitan los duplicados del reporte de recommend, y se remueve la columna count
valores_unigrams_unicos = df_uigrams_recommend.drop_duplicates(subset=['index','Recommend'])
valores_unigrams_unicos = valores_unigrams_unicos.drop('count',axis=1)
#Convertimos los valores de la tupla a string, y depues concatenamos
valores_unigrams_unicos["index"]=valores_unigrams_unicos["index"].astype(str)
valores_unigrams_unicos["index+Recommend"]=valores_unigrams_unicos["index"]+ valores_unigrams_unicos["Recommend"]
#Agarramos el df de las recomendaciones y hacemos el concatenado anterior
df_uigrams_recommend_modificado=df_uigrams_recommend
df_uigrams_recommend_modificado["index"]=df_uigrams_recommend_modificado["index"].astype(str)
df_uigrams_recommend_modificado["index+Recommend"]=df_uigrams_recommend_modificado["index"]+ df_uigrams_recommend_modificado["Recommend"]
#Hacemos un ciclo For para hacer la suma acumulada "Frecuencia Unigrama + Reseña"

valores_unigrams_unicos["Frecuencia Unigrama + Reseña"]= 0

colC=df_uigrams_recommend_modificado.columns.get_loc('count')

colFUR=valores_unigrams_unicos.columns.get_loc("Frecuencia Unigrama + Reseña")

loc=0

for gramm in valores_unigrams_unicos['index+Recommend']:
    cc=0
    loc2=0
    
    for gramm_2 in df_uigrams_recommend_modificado ['index+Recommend']:
        if gramm == gramm_2:
            cc = cc + df_uigrams_recommend_modificado.iloc[loc2,colC]
        loc2=loc2+1
    
    valores_unigrams_unicos.iloc[loc,colFUR]=cc
    loc=loc+1
#Hacemos un ciclo For para hacer la suma acumulada "Frecuencia Unigrama"

valores_unigrams_unicos["Frecuencia Unigrama"]=0
loc=0

colC=df_uigrams_recommend_modificado.columns.get_loc('count')
colFu=valores_unigrams_unicos.columns.get_loc('Frecuencia Unigrama')


for gramm in valores_unigrams_unicos['index']:
    cc=0
    loc2=0
    
    for gramm_2 in df_uigrams_recommend_modificado ['index']:
        if gramm == gramm_2:
            cc= cc+ df_uigrams_recommend_modificado.iloc[loc2,colC]
        loc2=loc2+1
    
    valores_unigrams_unicos.iloc[loc,colFu]=cc
    loc=loc+1
#Hacemos un ciclo For para calcular probabilidad

valores_unigrams_unicos["Probabilidad"]=None

colP=valores_unigrams_unicos.columns.get_loc("Probabilidad")
colFUR=valores_unigrams_unicos.columns.get_loc("Frecuencia Unigrama + Reseña")
colFU=valores_unigrams_unicos.columns.get_loc("Frecuencia Unigrama")

loc=0

for fecuencia in valores_unigrams_unicos['index']:

    valores_unigrams_unicos.iloc[loc,colP]=valores_unigrams_unicos.iloc[loc,colFUR]/valores_unigrams_unicos.iloc[loc,colFU]
    loc=loc+1

valores_unigrams_unicos.to_csv('ProbabilidadUniGramm.csv', index=False)

#Hacer tabla prob limpia

df_uigrams_test=pd.DataFrame({"Word":[],"V":[],"X":[],"O":[]})
df_uigrams_test['Word']=valores_unigrams_unicos['index'].drop_duplicates()

df_uigrams_test = df_uigrams_test.fillna(0)
colP=valores_unigrams_unicos.columns.get_loc("Probabilidad")
colRec=valores_unigrams_unicos.columns.get_loc("Recommend")
colV=df_uigrams_test.columns.get_loc("V")
colX=df_uigrams_test.columns.get_loc("X")
colO=df_uigrams_test.columns.get_loc("O")



locw=0
for word in df_uigrams_test['Word']:
    loci=0
    for index in valores_unigrams_unicos['index']:
        if word == index:
            if valores_unigrams_unicos.iloc[loci,colRec]=='v':
                df_uigrams_test.iloc[locw,colV]=valores_unigrams_unicos.iloc[loci,colP]
            if valores_unigrams_unicos.iloc[loci,colRec]=='x':
                df_uigrams_test.iloc[locw,colX]=valores_unigrams_unicos.iloc[loci,colP]
            if valores_unigrams_unicos.iloc[loci,colRec]=='o':
                df_uigrams_test.iloc[locw,colO]=valores_unigrams_unicos.iloc[loci,colP]
        loci=loci+1
    locw=locw+1 

df_uigrams_test.to_csv('ProbabilidadUniGramm_VXO.csv', index=False)

print("Listado de Probabilidades Unigram Train")
print(df_uigrams_test)

#Se abre el df test

df_glass_t_test=pd.read_csv("/Users/andresfelipe/Documents/Programacion-2/glassdor_test.csv")
df_glass_t_test.head(5)

#-------Se le da formato--------------


df_glass_t_test['headline'] = df_glass_t_test['headline'].fillna(",")
df_glass_t_test['pros'] = df_glass_t_test['pros'].fillna(",")
df_glass_t_test['cons'] = df_glass_t_test['cons'].fillna(",")

#Merge hedline, pros, cons
df_glass_t_test["headline+pros+cons"] = df_glass_t_test["headline"] + " " + df_glass_t_test["pros"]+ " " + df_glass_t_test["cons"]
df_glass_t_test["headline+pros+cons"]

#Make the info at column "headline+pros+cons" string
df_glass_t_test['headline+pros+cons']= df_glass_t_test['headline+pros+cons'].astype('string')

#Remove Punctuation Marks
# Apply function to the merged column
df_glass_t_test['headline+pros+cons'] = df_glass_t_test['headline+pros+cons'].apply(quitar_puntuacion)

#Remove Stop words
# Apply function to the merged column
df_glass_t_test['headline+pros+cons'] = df_glass_t_test['headline+pros+cons'].apply(quitar_stop)

#Lematize
# Aplicamos la lematización a cada fila
nlp = spacy.load("en_core_web_sm")
df_glass_t_test['headline+pros+cons_lemma'] = df_glass_t_test['headline+pros+cons'].apply(lematizar_)

#Hacer lemma string
df_glass_t_test['headline+pros+cons_lemma']=df_glass_t_test['headline+pros+cons_lemma'].astype('string')

#Remove Symbols Stop words
# Apply function to the merged column
df_glass_t_test['headline+pros+cons_cleanlemma'] = df_glass_t_test['headline+pros+cons_lemma'].apply(quitar_stop)

#Crea una columna para ubicar a usuario
df_glass_t_test.index.name = 'User'
#Index a columna
df_glass_t_test = df_glass_t_test.reset_index()
#Create new DF with unigrams,bigrams to record Y values (recommend)

colhpcl=df_glass_t_test.columns.get_loc('headline+pros+cons_cleanlemma')
colrec=df_glass_t_test.columns.get_loc('recommend')
colus=df_glass_t_test.columns.get_loc('User')


df_uigrams_test=pd.DataFrame({"count":[],"Recommend":[],"User":[]})
df_bigrams_test=pd.DataFrame({"count":[],"Recommend":[],"User":[]})


cc = 0

for x in df_glass_t_test['headline+pros+cons_cleanlemma']:

    text = df_glass_t_test.iloc[cc,colhpcl]
        
    tokens = nltk.word_tokenize(text)

    unigrams=list(ngrams(tokens, 1))
    bigrams=list(ngrams(tokens, 2))

    Ngrams=-1 #To record all

    unigram_counts = (pd.Series(unigrams).value_counts())[:Ngrams]
    bigram_counts = (pd.Series(bigrams).value_counts())[:Ngrams]

    df_new = unigram_counts.to_frame()
    df_newb = bigram_counts.to_frame()
    
    
    df_new["Recommend"]= df_glass_t_test.iloc[cc,colrec]
    df_new["User"]= df_glass_t_test.iloc[cc,colus]

    df_newb["Recommend"]= df_glass_t_test.iloc[cc,colrec]
    df_newb["User"]= df_glass_t_test.iloc[cc,colus]

    df_uigrams_test = pd.concat([df_new, df_uigrams_test])
    df_bigrams_test = pd.concat([df_newb, df_bigrams_test])

    cc = cc+ 1
#Index a columna
df_uigrams_test = df_uigrams_test.reset_index()
df_bigrams_test = df_bigrams_test.reset_index()

#Save the DF with unigrams test on a CSV
df_uigrams_test.to_csv('Unigrams_test.csv', index=False)
df_bigrams_test.to_csv('Bigrams_test.csv', index=False)

df_uigrams_test["index"]=df_uigrams_test["index"].astype(str)
df_bigrams_test["index"]=df_bigrams_test["index"].astype(str)

#Abre csvs Probabilidades

df_prob=pd.read_csv('ProbabilidadUniGramm_VXO.csv')
df_probb=pd.read_csv('ProbabilidadBiGramm_VXO.csv')

#Calcular Probabilidades Unigram Bigram

#Primero sacamos los usuarios unicos

usuarios_unicos=pd.DataFrame({"User":[],"ProbV":[],"ProbX":[],"ProbO":[]})
usuarios_unicosb=pd.DataFrame({"User":[],"ProbV":[],"ProbX":[],"ProbO":[]})


usuarios_unicos['User']=df_uigrams_test["User"].drop_duplicates()
usuarios_unicosb['User']=df_bigrams_test["User"].drop_duplicates()

#Quita Na

usuarios_unicos = usuarios_unicos.fillna(0)
usuarios_unicosb = usuarios_unicosb.fillna(0)

#Calcula numeradores Probabilidades Unigram
loccu=0

colUs=df_uigrams_test.columns.get_loc('User')
colCount=df_uigrams_test.columns.get_loc('count')

colPv=usuarios_unicos.columns.get_loc('ProbV')
colPx=usuarios_unicos.columns.get_loc('ProbX')
colPo=usuarios_unicos.columns.get_loc('ProbO')

colV=df_prob.columns.get_loc('V')
colX=df_prob.columns.get_loc('X')
colO=df_prob.columns.get_loc('O')

for user in usuarios_unicos['User']:
    locct=0
    for palabra_test in df_uigrams_test["index"]:
        if user == df_uigrams_test.iloc[locct,colUs]:
            loccp=0
            for pal_prob in df_prob ["Word"]:
                if pal_prob == palabra_test:
                    if df_prob.iloc[loccp,colV] != 0:
                        if usuarios_unicos.iloc[loccu,colPv]==0:
                            usuarios_unicos.iloc[loccu,colPv]=1
                        usuarios_unicos.iloc[loccu,colPv]= usuarios_unicos.iloc[loccu,colPv]*(df_prob.iloc[loccp,colV]**df_uigrams_test.iloc[locct,colCount])
                    
                    if df_prob.iloc[loccp,colX] != 0:
                        if usuarios_unicos.iloc[loccu,colPx]==0:
                            usuarios_unicos.iloc[loccu,colPx]=1
                        usuarios_unicos.iloc[loccu,colPx]= usuarios_unicos.iloc[loccu,colPx]*(df_prob.iloc[loccp,colX]**df_uigrams_test.iloc[locct,colCount])
                    
                    if df_prob.iloc[loccp,colO] != 0:
                        if usuarios_unicos.iloc[loccu,colPo]==0:
                            usuarios_unicos.iloc[loccu,colPo]=1
                        usuarios_unicos.iloc[loccu,colPo]= usuarios_unicos.iloc[loccu,colPo]*(df_prob.iloc[loccp,colO]**df_uigrams_test.iloc[locct,colCount])
                loccp=loccp+1
        locct=locct+1
    loccu=loccu+1
#Calcula numeradores Probabilidades Bigram
loccu=0

colUs=df_bigrams_test.columns.get_loc('User')
colCount=df_bigrams_test.columns.get_loc('count')

colPv=usuarios_unicosb.columns.get_loc('ProbV')
colPx=usuarios_unicosb.columns.get_loc('ProbX')
colPo=usuarios_unicosb.columns.get_loc('ProbO')

colV=df_probb.columns.get_loc('V')
colX=df_probb.columns.get_loc('X')
colO=df_probb.columns.get_loc('O')

for user in usuarios_unicosb['User']:
    locct=0
    for palabra_test in df_bigrams_test["index"]:
        if user == df_bigrams_test.iloc[locct,colUs]:
            loccp=0
            for pal_probb in df_probb ["Word"]:
                if pal_probb == palabra_test:
                    if df_probb.iloc[loccp,colV] != 0:
                        if usuarios_unicosb.iloc[loccu,colPv]==0:
                            usuarios_unicosb.iloc[loccu,colPv]=1
                        usuarios_unicosb.iloc[loccu,colPv]= usuarios_unicosb.iloc[loccu,colPv]*(df_probb.iloc[loccp,colV]**df_bigrams_test.iloc[locct,colCount])
                    
                    if df_probb.iloc[loccp,colX] != 0:
                        if usuarios_unicosb.iloc[loccu,colPx]==0:
                            usuarios_unicosb.iloc[loccu,colPx]=1
                        usuarios_unicosb.iloc[loccu,colPx]= usuarios_unicosb.iloc[loccu,colPx]*(df_probb.iloc[loccp,colX]**df_bigrams_test.iloc[locct,colCount])
                    
                    if df_probb.iloc[loccp,colO] != 0:
                        if usuarios_unicosb.iloc[loccu,colPo]==0:
                            usuarios_unicosb.iloc[loccu,colPo]=1
                        usuarios_unicosb.iloc[loccu,colPo]= usuarios_unicosb.iloc[loccu,colPo]*(df_probb.iloc[loccp,colO]**df_bigrams_test.iloc[locct,colCount])
                loccp=loccp+1
        locct=locct+1
    loccu=loccu+1
print("Cálculo de numeradores por Usuario Bigrama:")
print(usuarios_unicosb)

#Normalizacion para encontrar las probabilidades reales Unigrama

#Se saca el total de las probabilidades por usuario

usuarios_unicos["SumProb"]=None
#Cálculo probabilidad normalizada
usuarios_unicos["ProbNomV%"]=None
usuarios_unicos["ProbNomX%"]=None
usuarios_unicos["ProbNomO%"]=None


colPv=usuarios_unicos.columns.get_loc('ProbV')
colPx=usuarios_unicos.columns.get_loc('ProbX')
colPo=usuarios_unicos.columns.get_loc('ProbO')

colPvp=usuarios_unicos.columns.get_loc("ProbNomV%")
colPxp=usuarios_unicos.columns.get_loc("ProbNomX%")
colPop=usuarios_unicos.columns.get_loc("ProbNomO%")

colSum=usuarios_unicos.columns.get_loc('SumProb')

locs=0

for usuario in usuarios_unicos["SumProb"]:
    usuarios_unicos.iloc[locs,colSum]=usuarios_unicos.iloc[locs,colPv]+usuarios_unicos.iloc[locs,colPx]+usuarios_unicos.iloc[locs,colPo]
    locs=locs+1

locn=0

for usuario in usuarios_unicos["ProbNomV%"]:
    if usuarios_unicos.iloc[locn,colSum] !=0:
        usuarios_unicos.iloc[locn,colPvp]=round((usuarios_unicos.iloc[locn,colPv]/usuarios_unicos.iloc[locn,colSum])*100,2)
        usuarios_unicos.iloc[locn,colPxp]=round((usuarios_unicos.iloc[locn,colPx]/usuarios_unicos.iloc[locn,colSum])*100,2)
        usuarios_unicos.iloc[locn,colPop]=round((usuarios_unicos.iloc[locn,colPo]/usuarios_unicos.iloc[locn,colSum])*100,2)
    else:
        usuarios_unicos.iloc[locn,colPvp]=0
        usuarios_unicos.iloc[locn,colPxp]=0
        usuarios_unicos.iloc[locn,colPop]=0
    locn=locn+1

#Normalizacion para encontrar las probabilidades reales Bigrama

#Se saca el total de las probabilidades por usuario

usuarios_unicosb["SumProb"]=None

#Cálculo probabilidad normalizada
usuarios_unicosb["ProbNomV%"]=None
usuarios_unicosb["ProbNomX%"]=None
usuarios_unicosb["ProbNomO%"]=None

colPv=usuarios_unicosb.columns.get_loc('ProbV')
colPx=usuarios_unicosb.columns.get_loc('ProbX')
colPo=usuarios_unicosb.columns.get_loc('ProbO')

colPvp=usuarios_unicosb.columns.get_loc("ProbNomV%")
colPxp=usuarios_unicosb.columns.get_loc("ProbNomX%")
colPop=usuarios_unicosb.columns.get_loc("ProbNomO%")

colSum=usuarios_unicosb.columns.get_loc('SumProb')

locs=0

for usuario in usuarios_unicosb["SumProb"]:
    usuarios_unicosb.iloc[locs,colSum]=usuarios_unicosb.iloc[locs,colPv]+usuarios_unicosb.iloc[locs,colPx]+usuarios_unicosb.iloc[locs,colPo]
    locs=locs+1

locn=0

for usuario in usuarios_unicosb["ProbNomV%"]:
    if usuarios_unicosb.iloc[locn,colSum] !=0:
        usuarios_unicosb.iloc[locn,colPvp]=round((usuarios_unicosb.iloc[locn,colPv]/usuarios_unicosb.iloc[locn,colSum])*100,2)
        usuarios_unicosb.iloc[locn,colPxp]=round((usuarios_unicosb.iloc[locn,colPx]/usuarios_unicosb.iloc[locn,colSum])*100,2)
        usuarios_unicosb.iloc[locn,colPop]=round((usuarios_unicosb.iloc[locn,colPo]/usuarios_unicosb.iloc[locn,colSum])*100,2)
    else:
        usuarios_unicosb.iloc[locn,colPvp]=0
        usuarios_unicosb.iloc[locn,colPxp]=0
        usuarios_unicosb.iloc[locn,colPop]=0
    locn=locn+1

#Comparar resultados obtenidos vs train Unigram

usuarios_unicos['Answer Predicted']= ""

colPvp=usuarios_unicos.columns.get_loc("ProbNomV%")
colPxp=usuarios_unicos.columns.get_loc("ProbNomX%")
colPop=usuarios_unicos.columns.get_loc("ProbNomO%")

colAnp=usuarios_unicos.columns.get_loc('Answer Predicted')

locap=0
for user in usuarios_unicos['User']:
    xVal=usuarios_unicos.iloc[locap,colPxp]
    vVal=usuarios_unicos.iloc[locap,colPvp]
    oVal=usuarios_unicos.iloc[locap,colPop]

    if xVal > vVal and xVal > oVal:
        usuarios_unicos.iloc[locap,colAnp]='x'
    if vVal > xVal and vVal > oVal:
        usuarios_unicos.iloc[locap,colAnp]='v'
    if oVal > vVal and oVal > xVal:
        usuarios_unicos.iloc[locap,colAnp]='o'
    locap=locap+1

#Comparar resultados obtenidos vs train Bigram

usuarios_unicosb['Answer Predicted']= ""

colPvp=usuarios_unicosb.columns.get_loc("ProbNomV%")
colPxp=usuarios_unicosb.columns.get_loc("ProbNomX%")
colPop=usuarios_unicosb.columns.get_loc("ProbNomO%")

colAnp=usuarios_unicosb.columns.get_loc('Answer Predicted')

locap=0
for user in usuarios_unicosb['User']:
    xVal=usuarios_unicosb.iloc[locap,colPxp]
    vVal=usuarios_unicosb.iloc[locap,colPvp]
    oVal=usuarios_unicos.iloc[locap,colPop]

    if xVal > vVal and xVal > oVal:
        usuarios_unicosb.iloc[locap,colAnp]='x'
    if vVal > xVal and vVal > oVal:
        usuarios_unicosb.iloc[locap,colAnp]='v'
    if oVal > vVal and oVal > xVal:
        usuarios_unicosb.iloc[locap,colAnp]='o'
    locap=locap+1

usuarios_unicos.to_csv('Results_test_Unigram.csv', index=False)
usuarios_unicosb.to_csv('Results_test_Bigram.csv', index=False)

#Concatenar respuestas presichas y reales Unigrama
usuarios_unicos['Real Answer']=""

colRA=usuarios_unicos.columns.get_loc("Real Answer")

colRm=df_uigrams_test.columns.get_loc("Recommend")


locus=0
for user in usuarios_unicos["User"]:
    locni=0
    for useruni in df_uigrams_test["User"]:
        if user == useruni:
            usuarios_unicos.iloc[locus,colRA]=df_uigrams_test.iloc[locni,colRm]
        locni=locni+1
    locus=locus+1

#Concatenar respuestas presichas y reales Bigrama


usuarios_unicosb['Real Answer']=""

colRA=usuarios_unicosb.columns.get_loc("Real Answer")

colRm=df_bigrams_test.columns.get_loc("Recommend")


locus=0
for user in usuarios_unicosb["User"]:
    locni=0
    for useruni in df_bigrams_test["User"]:
        if user == useruni:
            usuarios_unicosb.iloc[locus,colRA]=df_bigrams_test.iloc[locni,colRm]
        locni=locni+1
    locus=locus+1

y_pred=usuarios_unicos["Answer Predicted"]
y_test=usuarios_unicos["Real Answer"]

y_predb=usuarios_unicosb["Answer Predicted"]
y_testb=usuarios_unicosb["Real Answer"]

print("Predicciones y Probabilidades Unigramas:")
print(usuarios_unicos)
print("Predicciones y Probabilidades Bigramas:")
print(usuarios_unicosb)

tP=0
sUM=0
cc=0

for x in y_pred:
    sUM=sUM+1
    if x == y_test.iloc[cc]:
        tP = tP + 1
    cc=cc+1

tPb=0
sUMb=0
ccb=0

for x in y_predb:
    sUMb=sUMb+1
    if x == y_testb.iloc[ccb]:
        tPb = tPb + 1
    ccb=ccb+1

print(f'Accuracy Unigrama: {tP/sUM}')
print(f'Accuracy Bigrama: {tPb/sUMb}')

mlflow.set_experiment("Deteccion de opiniones Empleados")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
with mlflow.start_run(run_name='Model_Unigram'):
    mlflow.log_param('model_name','Model_Unigram')
    mlflow.log_metric('accuracy',tP/sUM)

with mlflow.start_run(run_name='Model_Bigram'):
    mlflow.log_param('model_name','Model_Bigram')
    mlflow.log_metric('accuracy',tPb/sUMb)