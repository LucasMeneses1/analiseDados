#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns

dataset_original = pd.read_csv('BankChurners.csv')
#display(dataset_original)

dataset_colunas_excluidas = dataset_original.drop(["CLIENTNUM", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], axis=1)
#display(dataset_colunas_excluidas)

df_traducao = pd.read_excel('TraducaoColunas.xlsx')
#display(df_traducao)
colunas_originais = list(dataset_colunas_excluidas.columns)
colunas_traduzidas = list(df_traducao['Tradução'])
dic_traducao = dict(zip(colunas_originais, colunas_traduzidas))
dataset_colunas_excluidas = dataset_colunas_excluidas.rename(columns = dic_traducao)
#display(dataset_colunas_excluidas)

dataset_tratado = dataset_colunas_excluidas.copy()
# Análise preliminar
#display(dataset_tratado.describe())
#display(dataset_tratado.describe(include=['object']))
# Conferindo a quantidade de clientes ativos e cancelados (valores absolutos)
#display(dataset_tratado['Status_Cliente'].value_counts())
# Conferindo a quantidade de clientes ativos e cancelados (valores em porcentagem)
#display(dataset_tratado['Status_Cliente'].value_counts(normalize=True))


# In[4]:


###################################################################################################
fig = px.histogram(dataset_tratado, 
                       x='Status_Cliente', 
                       color='Status_Cliente',  
                       color_discrete_sequence=['#4682B4', '#FF8C00'],
                       text_auto=True,
                       labels={'Status_Cliente': 'Status do Cliente'}
                  )
fig.add_annotation( 
    text="<b>16,07%</b> do total de clientes", x=1, y=1680, arrowhead=1, showarrow=True
)
                           
fig.show()
###################################################################################################
fig = px.histogram(dataset_tratado, 
                       x='Categoria_Cartão', 
                       color='Status_Cliente',
                       text_auto=True,
                       color_discrete_sequence=['#4682B4', '#FF8C00'],
                       labels={'Status_Cliente': 'Status do Cliente', 'Categoria_Cartão':'Categoria do Cartão'})

fig.add_annotation( 
    text="A categoria Blue representa <br> <b>93,2%</b> do total de clientes", x=0.05, y=9700, arrowhead=1, showarrow=True
)
                           
fig.show()
###################################################################################################
fig = px.histogram(dataset_tratado, 
                       x='Qtde_Transacoes_12m', 
                       color='Status_Cliente',  
                       color_discrete_sequence=['#4682B4', '#FF8C00'],
                       labels={'Status_Cliente': 'Status do Cliente', "Qtde_Transacoes_12m": "Quantidade de transações nos últimos 12 meses"})
                           
fig.show()
###################################################################################################
fig = px.histogram(dataset_tratado, 
                       x='Valor_Transacoes_12m', 
                       color='Status_Cliente',  
                       color_discrete_sequence=['#4682B4', '#FF8C00'],
                       labels={'Status_Cliente': 'Status do Cliente', "Valor_Transacoes_12m": "Valor das transações dos últimos 12 meses"})
                           
fig.show()
###################################################################################################
fig = px.histogram(dataset_tratado, 
                       x='Limite_Consumido', 
                       color='Status_Cliente',  
                       color_discrete_sequence=['#4682B4', '#FF8C00'],
                       labels={'Status_Cliente': 'Status do Cliente', "Limite_Consumido": "Limite Consumido"})
                           
fig.show()
###################################################################################################
fig = px.histogram(dataset_tratado, 
                       x='Produtos_Contratados', 
                       color='Status_Cliente',  
                       color_discrete_sequence=['#4682B4', '#FF8C00'],
                       labels={'Status_Cliente': 'Status do Cliente', "Produtos_Contratados": "Produtos Contratados"})
                           
fig.show()
###################################################################################################
fig = px.histogram(dataset_tratado, 
                       x='Contatos_12m', 
                       color='Status_Cliente',  
                       color_discrete_sequence=['#4682B4', '#FF8C00'],
                       labels={'Status_Cliente': 'Status do Cliente', "Contatos_12m": "Contatos nos últimos 12 meses"})
                           
fig.show()


# In[5]:


for feature in dataset_tratado:
    fig = px.histogram(dataset_tratado, 
                       x=feature, 
                       color='Status_Cliente',  
                       color_discrete_sequence=['#4682B4', '#FF8C00'],
                       labels={'Status_Cliente': 'Status do Cliente'})
                           
    fig.show()

