#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: john
"""
import pandas as pd
import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

#Leitura dos dados

filmes = pd.read_csv("datasets/avaliacoes_usuario.csv")

#Separação dos atributos e marcações e, então, separamos em dados de treino e teste

filmes_atributos = filmes[filmes.columns[1:16]]
filmes_gostou = filmes[filmes.columns[16:]]
treino, teste, treino_marcacoes, teste_marcacoes =  train_test_split(filmes_atributos, filmes_gostou)

"""
Organizando os dados para aplicar a regressão
    Precisamos transformar o nosso vetor de treino em um vetor coluna e, para isso, utilizamos a funcao reshape e array do numpy.
"""
treino = np.array(treino).reshape(len(treino), 15)
teste = np.array(teste).reshape(len(teste), 15)

#O vetor coluna do treino_marcacoes precisa ser convertido em um array

treino_marcacoes = treino_marcacoes.values.ravel()
teste_marcacoes = teste_marcacoes.values.ravel()

#Criando o modelo

modelo = LogisticRegression()
modelo.fit(treino,treino_marcacoes)

#previsao de dados
previsoes =  modelo.predict(teste)

acuracia = accuracy_score(teste_marcacoes, previsoes)
acuracia

#NaiveBayes

modelo_NB = GaussianNB()
modelo_NB.fit(treino, treino_marcacoes)
previsoes_NB =  modelo_NB.predict(teste)
acuracia_NB = accuracy_score(teste_marcacoes, previsoes_NB)
acuracia_NB
    