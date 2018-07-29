#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: john
"""
import pandas as pd
import numpy as np 
from sklearn.linear_model  import LinearRegression
from sklearn.model_selection import train_test_split

filmes = pd.read_csv("datasets/movies_multilinear_reg.csv")

#separações
filmes_atributos = filmes[filmes.columns[2:17]]
filmes_bilheteria = filmes[filmes.columns[17:]]
treino, teste, treino_marcacoes, teste_marcacoes =  train_test_split(filmes_atributos, filmes_bilheteria)

#Organizando os dados para aplicar a regressão
treino = np.array(treino).reshape(len(treino), 15)
teste = np.array(teste).reshape(len(teste), 15)

#Criando o modelo
modelo = LinearRegression()
modelo.fit(treino,treino_marcacoes)

#calculando o r²
modelo.score(treino,treino_marcacoes)
modelo.score(teste,teste_marcacoes)

#testando com casos reais
zootopia = [0,0,0,0,0,0,0,0,1,1,1,0,1,145.5170642,3.451632127]
modelo.predict([zootopia])