import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

df1 = pd.read_csv('/content/campeonato-brasileiro-estatisticas-full.csv')
df2 = pd.read_csv('/content/campeonato-brasileiro-full.csv')

lista = []
for x in df1['escanteios']:
    if x >5:
        x = 1
        lista.append(x)
    elif x<=5:
        x = 0
        lista.append(x)

df1 = df1.assign(escanteio=lista)

#Modelagem de dados

df1.drop('partida_id', axis=1, inplace=True)
df1.info()
df1.drop(['rodada', 'clube', 'escanteios'], axis=1, inplace=True)
#df1 = df1.dropna()
#print(len(df1))

df1 = df1.mask(df1.eq('None')).dropna()
#print(df1.columns)

base_x= df1.iloc[:,:9].values
#print(base_x[0])
base_y = df1.iloc[:,9].values
#print(base_y.shape, base_x[:,0])

base_x[:,0] = label_encoder_chutes.fit_transform(base_x[:,0])
base_x[:,1] = label_encoder_chutes_no_alvo.fit_transform(base_x[:,1])
base_x[:,2] = label_encoder_posse_de_bola.fit_transform(base_x[:,2])
base_x[:,3] = label_encoder_passes.fit_transform(base_x[:,3])
base_x[:,4] = label_encoder_precisao_passes.fit_transform(base_x[:,4])
base_x[:,5] = label_encoder_faltas.fit_transform(base_x[:,5])
base_x[:,6] = label_encoder_cartao_amarelo.fit_transform(base_x[:,6])
base_x[:,7] = label_encoder_cartao_vermelho.fit_transform(base_x[:,7])
base_x[:,8] = label_encoder_impedimentos.fit_transform(base_x[:,8])

onehotencoder_base_x = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(),
                                                       [0,1,2,3,4,5,6,7,8])], remainder='passthrough')

base_x = onehotencoder_base_x.fit_transform(base_x).toarray()
scaler_x = StandardScaler()
base_x = scaler_x.fit_transform(base_x)

import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

base_x_treinamento, base_x_teste, base_y_treinamento, base_y_teste = train_test_split(base_x, base_y,
                                                                                  test_size=0.15, random_state=0)

#print(base_y_treinamento.shape)

with open('picklefut.pkl', mode = 'wb') as f:
    pickle.dump([base_x_treinamento, base_y_treinamento, base_x_teste, base_y_teste], f)

with open('picklefut.pkl', 'rb') as f:
    base_x_treinamento, base_y_treinamento, base_x_teste, base_y_teste = pickle.load(f)

#print(base_y_treinamento.shape)

naive_fut_data = GaussianNB()
naive_fut_data.fit(base_x_treinamento, base_y_treinamento)

#Criar a variável para receber os dados
previsoes = naive_fut_data.predict(base_x_teste)

#apurando resultados
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix

#print(accuracy_score(base_y_teste, previsoes))

cm = ConfusionMatrix(naive_fut_data)
cm.fit(base_x_treinamento, base_y_treinamento)
#print(cm.score(base_x_teste, base_y_teste))
