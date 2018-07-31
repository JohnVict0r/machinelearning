#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: john
"""
import pandas as pd
import numpy as np 
from sklearn.linear_model  import LinearRegression
from sklearn.model_selection import train_test_split

#Leitura dos dados
filmes = pd.read_csv("datasets/regressao_linear_alura.csv")

#Separação dos dados nas variáveis dependente e independente

filmes_investimento = filmes['Investimento (em milhoes)']
filmes_bilheteria = filmes['Bilheteria (pessoas)']

#Separação dos dados em treino e teste (O split padrão é 0.25)

treino, teste, treino_marcacoes, teste_marcacoes =  train_test_split(filmes_investimento, filmes_bilheteria)

"""
Reorganização dos dados

    Precisamos transformar o nosso vetor de treino em um vetor coluna e, para isso, utilizamos a função reshape e array do numpy
    Passando o primeiro parâmetro como o numero do nosso array e o segundo o numero de colunas (no caso é 1)
"""
treino = np.array(treino).reshape(len(treino), 1)
teste = np.array(teste).reshape(len(teste), 1)

#Criação do modelo

modelo = LinearRegression()
modelo.fit(treino,treino_marcacoes)

#modelo.predict([[129012]])

#Pegar o r quadrado da nossa funcao linear

modelo.score(treino,treino_marcacoes)

#Pegar o r quadrado da nossa funcao linear para os teste

modelo.score(teste,teste_marcacoes)

#testando com casos reais
zootopia = [0,0,0,0,0,0,0,0,1,1,1,0,1,145.5170642,3.451632127]
modelo.predict([zootopia])

planet_apes = [0,1,0,0,0,0,0,0,0,0,0,0,0,150,5]
modelo.predict([planet_apes])



