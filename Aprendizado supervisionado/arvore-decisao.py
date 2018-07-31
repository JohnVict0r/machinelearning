#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: john
"""
import pandas as pd
import numpy as np 
from sklearn  import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

filmes = pd.read_csv("datasets/movies_multilinear_reg.csv")

filmes_atributos = filmes[filmes.columns[2:17]]
filmes_bilheteria = filmes[filmes.columns[17:]]
treino, teste, treino_marcacoes, teste_marcacoes =  train_test_split(filmes_atributos, filmes_bilheteria)


treino = np.array(treino).reshape(len(treino), 15)
teste = np.array(teste).reshape(len(teste), 15)

#Criando e avaliando os modelos
zootopia = [0,0,0,0,0,0,0,0,1,1,1,0,1,145.5170642,3.451632127]
#modelo 1

model = tree.DecisionTreeRegressor(max_depth=5)
model.fit(treino,treino_marcacoes)

model.score(treino,treino_marcacoes)
model.score(teste,teste_marcacoes)

model.predict([zootopia])

#modelo 2
modelo = tree.DecisionTreeClassifier(max_depth=5)
lab_enc = preprocessing.LabelEncoder()
treino_marcacoes = lab_enc.fit_transform(treino_marcacoes).ravel()
modelo.fit(treino,treino_marcacoes)

teste_marcacoes = lab_enc.fit_transform(teste_marcacoes)

modelo.score(treino,treino_marcacoes)
modelo.score(teste,teste_marcacoes)

modelo.predict([zootopia])





