#!/usr/bin/env python
# coding: utf-8

# # 1. Desafio e Dataset
# <br>
# - O diretor de uma grande empresa de Cartão de Crédito percebeu que o número de clientes que cancelam seus cartões tem aumentado significativamente, causando prejuízos enormes para a empresa.
# <br>
# <br>
# Nosso objetivo é descobrir o motivo de os clientes cancelarem o cartão. Analisaremos o dataset fornecido para tentar descobrir quais pessoas têm maior tendência de solicitar o cancelamento do cartão e o que fazer para evitar que isso aconteça.
# <br>
# <br>
# - Referência: Tanto o problema apresentado quanto o dataset utilizado foram retirados do site Kaggle (https://www.kaggle.com/sakshigoyal7/credit-card-customers)

# # 2. Tratando o dataset

# In[14]:


# Importando o dataset
import pandas as pd

dataset_original = pd.read_csv('BankChurners.csv')
display(dataset_original)


# - Observando as colunas, de início percebemos que a coluna "CLIENTNUM" exibe apenas o código de representação de cada cliente. Logo, não será necessária para a análise e será excluida.
# <br>
# <br>
# <br>
# - Outras colunas que podemos excluir de início são as colunas "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1" e "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2". <br><br> A exclusão dessas colunas é uma recomendação do autor do desafio, pois não fornecem nenhuma informação relevante para a análise.

# In[15]:


# Excluindo as colunas
dataset_colunas_excluidas = dataset_original.drop(["CLIENTNUM", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"], axis=1)
display(dataset_colunas_excluidas)


# - dados faltantes

# In[ ]:





# - As colunas restantes são colunas que fornecem informações diversas sobre os clientes. Uma vez que todas elas podem ter algum tipo de influência sobre a decisão de permanencia ou não como cliente do banco, não excluiremos mais nenhuma coluna.

# # 3. Realizando análises no dataset tratado

# - Agora que tratamos o dataset, iremos realizar uma análise exploratória para conhecer melhor os dados, tentar encontrar algum padrão no comportamento deles e retirar alguma conclusão que nos aproxime da resposta para o problema proposto. Para realizar nossa análise exploratória, utilizaremos histogramas criados a partir do nosso dataset. Para facilitar o acompanhamento, vamos traduzir os nomes das colunas.

# In[16]:


# Tradução das colunas
df_traducao = pd.read_excel('TraducaoColunas.xlsx')
display(df_traducao)
colunas_originais = list(dataset_colunas_excluidas.columns)
colunas_traduzidas = list(df_traducao['Tradução'])
dic_traducao = dict(zip(colunas_originais, colunas_traduzidas))
dataset_colunas_excluidas = dataset_colunas_excluidas.rename(columns = dic_traducao)
display(dataset_colunas_excluidas)


# In[17]:


dataset_tratado = dataset_colunas_excluidas.copy()
# Análise preliminar
display(dataset_tratado.describe())
display(dataset_tratado.describe(include=['object']))
# Conferindo a quantidade de clientes ativos e cancelados (valores absolutos)
display(dataset_tratado['Status_Cliente'].value_counts())
# Conferindo a quantidade de clientes ativos e cancelados (valores em porcentagem)
display(dataset_tratado['Status_Cliente'].value_counts(normalize=True))


# In[ ]:





# - Acima temos algumas informações gerais acerca da nossa base de clientes. Por meio do método describe obtemos, separadamente, alguns parâmetros das features numéricas e textuais do nosso dataset. Alguns pontos que podem ser importantes para a análise são fornecidos por este método, como por exemplo a média de idade dos clientes, a média da quantidade de produtos que os clientes adquiriram do banco, o intervalo de limite de cartão de crédito que concentra a maior parte dos clientes, o nível de escolaridade mais presente entre os clientes, dentre outros.
# <br>
# <br>
# - Além do método describe, utilizamos o método value_counts para observar como a coluna de Status_Cliente (nosso alvo) está dividida. Com esse método, pudemos observar que, dentre os 10127 clientes presentes no dataset, cerca de 16% deixou o banco.
# <br>
# <br>
# <h5>Lembrando que na coluna 'Status_Cliente', o valor 'Existing Customer' significa cliente com cartão ativo, e o valor 'Attrited Customer' significa cliente que cancelou seu cartão.<h5>

# In[18]:


# Gerando os histogramas
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns

for feature in dataset_tratado:
    fig = px.histogram(dataset_tratado, x=feature, color='Status_Cliente', color_discrete_sequence=['#4682B4', '#FF8C00'])            
    fig.show()


# - O objetivo dessa análise exploratória é encontrar alguma característica ou algum conjunto de características que formem um perfil dos clientes alvo. Utilizando os conceitos da regra de Pareto, buscaremos encontrar essas características que são comuns entre a maior parte dos clientes que cancelam seu cartão para podermos trabalhar em soluções em cima delas.

# # 4. Conclusões
# <br>
# <br>
# - Observando o gráfico da coluna "Categoria Cartão", observamos que os cancelamentos de clientes da categoria blue representam quase que a totalidade dos cancelamentos.
# <br>
# <br>
# - Observando o gráfico da coluna "Contatos", percebemos que a proporção de cancelamentos cresce a medida que mais contatos são realizados, ou seja, quanto mais o cliente entra em contato com o banco, maior a chance de ele cancelar o cartão. -> analizar o motivo do contato
# <br>
# <br>
# - Observando o gráfico da coluna "Limite" (limite do cartão), observamos que a maior concentração de cancelamentos está no intervalo de clientes com limite de até 6.000 reais, com destaque para a taxa de cancelamentos entre clientes com menos de 4.250 reais. Ou seja, clientes com pouco limite disponível tem grandes chances de realizar o cancelamento. Esse comportamento é reforçado pelo gráfico da coluna "Limite Disponível".
# <br>
# <br>
# - Observando o gráfico da coluna "Qtd Transações" (quantidade de transações), percebemos que a maior concentração de clientes que cancelaram o cartão está na faixa de 0-55 transações, mesmo com a maior concetração de clientes totais estar na faixa de 54-130 transações. Ou seja , temos um indicativo muito forte de que a quantidade de transações é uma das caracteristícas principais do perfil de cliente que cancela seu cartão. Esse comportamento é reforçado pelos gráficos das colunas "Valor Transações 12m" (valor das transações realizadas nos últimos 12 meses), pela coluna "Taxa de utilização do cartão" e pela coluna "Limite Consumido". 
# <br>
# <br>
# <br>
# <br>
# 
# 
# A partir de 60 transações, a taxa de cancelamentos cai drasticamente. Clientes com menos de 60 transações são clientes críticos, com alta probabilidade de realizar cancelamento. 

# # 5. Complementando a análise com machine learning

# - Buscando validar a análise exploratória e encontrar alguma feature que acrescente na análise do problema, aplicaremos algoritmos de machine learning para tertarmos criar um modelo de previsão para a evasão de clientes.
# <br>
# <br>
# - Ao encontrarmos um modelo de previsão bom o suficiente, analisaremos como cada feature contribuiu dentro do modelo para a obtenção do resultado final e, assim, encontrar quais features são as mais importantes dentro do perfil de cliente que cancela seu cartão.

# In[19]:


# Preparando o dataset para os tratamentos necessários para a aplicação de machine learning
dataset_ml = dataset_tratado.copy()
display(dataset_ml.head(10))
display(dataset_ml.info())


# # 6. Encoding

# - Uma vez que os algoritmos de machine learning que serão utilizados neste projeto não trabalham com variáveis textuais, precisaremos de alguma forma transformar essas variáveis textuais em variáveis numéricas. Esse processo é chamado de Encoding.
# <br>
# <br>
# - Antes de adaptá-las para uma forma numérica, vamos explorar como estão distribuídos os valores dessas colunas para, se necessário, simplificá-las de alguma forma. Se encontrar-mos colunas que possuam um conjunto de valores com quantidade insignificante em relação ao total, poderemos agrupá-las de alguma forma, ou excluí-las, dependendo da sua relevância para a análise.

# In[20]:


lista_colunas_cat = ['Sexo', 'Educação', 'Estado_Civil', 'Faixa_Salarial_Anual', 'Categoria_Cartão']

plt.figure(figsize=(15, 5))
sns.countplot(x='Status_Cliente', data=dataset_ml)
    
for coluna in lista_colunas_cat:
    plt.figure(figsize=(15, 5))
    sns.countplot(x=coluna, data=dataset_ml)


# - Observando os gráficos das colunas, não encontramos valores que se necessitassem de tratamento. Logo, podemos seguir para a conversão dos valores textuais para valores numéricos.
# <br>
# <br>
# - A coluna que abriga os clientes alvo (Status_Cliente) é uma coluna do tipo verdadeiro/falso, no qual o cliente que cancelou seria um valor 'verdadeiro', e o cliente ativo seria um valor 'falso'. Sua convesão para coluna numérica será simples, sendo necessário apenas converter os seus valores para 0 nos valores falsos e 1 nos valores verdadeiros.
# <br>
# <br>
# - Ja para as demais colunas, que são colunas categóricas, aplicaremos a tecnica OneHot Encoding.

# In[21]:


dataset_enc = dataset_ml.copy()

# Realizando a conversão da coluna 'Status_Cliente'
dataset_enc.loc[dataset_enc['Status_Cliente'] == 'Attrited Customer', 'Status_Cliente'] = 1
dataset_enc.loc[dataset_enc['Status_Cliente'] == 'Existing Customer', 'Status_Cliente'] = 0

dataset_enc['Status_Cliente'] = dataset_enc['Status_Cliente'].astype(int)

# Conferindo a conversão
dataset_enc.loc[:,'Status_Cliente']


# In[22]:


# Realizando a conversão das demais colunas
dataset_enc = pd.get_dummies(data=dataset_enc, columns=lista_colunas_cat)

# Conferindo as conversões
display(dataset_enc.info()) # -> vemos que agora todas as colunas são numéricas
display(dataset_enc.iloc[0,:])


# # 7. Escolhendo o modelo de machine learning

# - Para criar nosso modelo de previsão, escolhemos testar os seguintes modelos: RandomForestClassifier e LogisticRegression.
# <br>
# <br>
# - Após realizar o treinamento dos modelos, escolheremos um deles para trabalhar baseado na análise dos seus parâmetros de desempenho.
# <br>
# <br>
# - Separaremos nosso dataset em dados de treino e dados de teste. Usaremos os dados de treino para treinar os algoritmos escolhidos de machine learning e avaliar os seus desempenhos. O treinamento de cada algoritmo será feito com validação cruzada. Dessa forma, os dados de treino serão separados em varios grupos (folds) de treino e validação, que permitirão aos modelos serem treinados com diferentes grupos de dados, garantido uma maior confiabilidade sobre os parâmetros de desempenho de cada modelo.
# <br>
# <br>
# - Utilizaremos como avaliadores de desempenho os seguintes parâmetro: Acurácia, Precisão, Recall e Matriz de Confusão.

# In[23]:


from sklearn.model_selection import train_test_split # Divide os dados em dados de treino e dados de teste
from sklearn.model_selection import cross_val_score, cross_val_predict # realiza a validação cruzada
from sklearn.model_selection import StratifiedKFold # separa os dados em folds com a msm proporção de casos da variavel de interesse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # modelo de previsão 1
from sklearn.linear_model import LogisticRegression # modelo de previsão 2
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, confusion_matrix

dataset_tratado = dataset_enc.copy()
# Separação das variáveis
y = dataset_tratado['Status_Cliente']
X = dataset_tratado.drop('Status_Cliente', axis=1)

# Semente de aleatoriedade
SEED = 14
np.random.seed(SEED)

# Separação dos dados de teste e dos dados de treino
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# In[24]:


# Preparando os folds
cv = StratifiedKFold(n_splits = 5, shuffle = True)

# Lista de modelos
model_rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model_lr = LogisticRegression(solver='liblinear')
model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
models = {
    'Random_Forest_Classifier': model_rfc,
    'Logistic_Regression': model_lr,
    "Gradient_Boosting_Classifier": model_gb
}

# Treinando e avaliando os modelos
for model_name, model in models.items():
    print('----------------------------------------------------------------')
    # Treinando e avaliando a acurácia
    results = cross_val_score(model, X_train, y_train, cv = cv, scoring = 'accuracy')
    mean = results.mean()
    dv = results.std()
    print('Acurácia média do modelo {}: {:.2f}%'.format(model_name, mean*100))
    print('Intervalo de acurácia do modelo {}: [{:.2f}% ~ {:.2f}%]'.format(model_name, (mean - 2*dv)*100, (mean + 2*dv)*100))
    print('\n')

    # Treinando e avaliando a precisão
    results = cross_val_score(model, X_train, y_train, cv = cv, scoring = 'precision')
    mean = results.mean()
    dv = results.std()
    print('Precisão média do modelo {}: {:.2f}%'.format(model_name, mean*100))
    print('Intervalo de precisão do modelo {}: [{:.2f}% ~ {:.2f}%]'.format(model_name, (mean - 2*dv)*100, (mean + 2*dv)*100))
    print('\n')
    
    # Treinando e avaliando o recall
    results = cross_val_score(model, X_train, y_train, cv = cv, scoring = 'recall')
    mean = results.mean()
    dv = results.std()
    print('Recall médio do modelo {}: {:.2f}%'.format(model_name, mean*100))
    print('Intervalo de recall do modelo {}: [{:.2f}% ~ {:.2f}%]'.format(model_name, (mean - 2*dv)*100, (mean + 2*dv)*100))                 
    print('\n')
    
    # Avaliando a matriz de confusão
    predicts = cross_val_predict(model, X_train, y_train, cv = cv)
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_train, predicts), annot=True, ax=ax, fmt='d', cmap='Greens')
    ax.set_title(f"Matriz de confusão do modelo {model_name}", fontsize=18)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    
    print('\n')


# - Analisando os parâmetros de desempenho, vemos que o algoritmo GradientBoostingClassifier gerou o modelo de melhor desempenho. Com isso, ele será o utilizado para gerar o modelo para estudo.

# # 8. Aplicando o modelo escolhido

# In[25]:


# Criando o modelo
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
gbc.fit(X_train, y_train)
predict = gbc.predict(X_test)

print('Precisão: ', precision_score(y_test, predict))
print('Recall: ', recall_score(y_test, predict))
print('F1 score: ', f1_score(y_test, predict))
print('MC: ', confusion_matrix(y_test, predict))
fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, predict), annot=True, ax=ax2, fmt='d', cmap='Blues')
ax2.set_title(f"Matriz de confusão do modelo Gradient Boosting Classifier", fontsize=15)
ax2.set_ylabel("True Label")
ax2.set_xlabel("Predicted Label")
plt.tight_layout()


# - Após gerar o modelo, podemos analisar o grau de importância de cada feature para o calculo final da previsão. Com isso, saberemos as features mais importantes, veremos se as nossas conclusões durante a análise exploratória foram corretas e poderemos chegar a novas conclusões observando características que podem ter passado desapercebidas durante a análise exploratória.

# In[26]:


# Gerando o gráfico de importância das colunas
importancia_features = pd.DataFrame(gbc.feature_importances_, X_train.columns).sort_values(by=0, ascending=False)
plt.figure(figsize=(25, 20))
sns.set(font_scale = 2)
ax = sns.barplot(x=importancia_features[0], y=importancia_features.index, orient='h')


# # 9. Conclusões da análise com machine learning

# - Como podemos ver no gráfico acima, o modelo teve como principais features que influenciaram na decisão do cliente de cancelar o cartão aquelas relacionadas ao volume de transações realizadas pelos clientes, corroborando o que nós já haviamos identificado na análise exploratória. 
# <br>
# <br>
# - Em seguida temos: o limite consumido, a quantidade de produtos contratados, e a quantidade de vezes que o cliente entrou em contato com o banco com significativa influência na decisão do cliente. A quantidade de produtos contratados foi uma característica que não identificamos na análise exploratória, mas que se mostrou significativa para a previsão do modelo.  

# # 10. Resultados e soluções propostas

# - Baseado nos resultados das análises, temos que o perfil de cliente com maior probabilidade de cancelar o cartão são: aqueles que usam pouco o cartão (principal característica), que possuem poucos produtos do banco e, que quando precisam entrar em contato com o banco, não conseguem resolver o problema nas primeiras ligações.
# <br>
# <br>
# - A partir do perfil encontrado, temos algumas sugestões de ações que podem ser aplicadas para conter o cancelamento dos clientes:
# <br>
# <br>
#  1. Precisamos incentivar o uso do cartão. Desenvolver sistemas de benefícios para quem utiliza o cartão do banco, como por exemplo um programa de pontos, é uma possível solução para esse problema. A grande maioria dos clientes são da categoria de cartão "Blue" (93% do total). Consequentemente, a grande maioria dos clientes que cancelam seus cartões estão nessa categoria. Desenvolver um sistema de benefícios aprimorado para o cartão "Blue" teria grandes chances de reduzir significativamente a evasão dos clientes.
# <br>
# <br>
#  2. Melhorar o serviço de atendimento ao cliente para conseguir-mos resolver o problema do cliente logo nas primeiras vezes que ele entrar em contato. Poderiamos também criar algum sistema de alerta para saber quando algum cliente entrou em contato com o banco mais de uma vez. Ao identificar esse tipo de cliente, o banco poderia priorizar o atendimento dele, conferindo se o seu problema ja foi resolvido e, caso não tenha sido resolvido, entrando em contato com ele para solucionar o problema em questão.
# <br>
# <br>
#  3. Facilitar o acesso aos produtos do banco. Constatamos que quanto mais produtos do banco o cliente tem, menor a chance de ele cancelar o cartão. Aproveitar a aplicação do item 1 e incluir nos benefícios um acesso mais fácil a produtos do banco pode ser uma boa solução, inclusive ajudando a resolver dois problemass de uma vez só. 
