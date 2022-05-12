#
#BIKE SHARING DATA SET
#
#Descrizione del Dataset al seguente link: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
#

import pandas as pd

#Colonne Dataset

headers=['instant','dteday','season','yr','mnth','holiday','weekday','workingday',
         'weathersit','temp','atemp','hum','windspeed','casual','registered','cnt']

head=['INSTANT','DTEDAY','SEASON','YR','MNTH','HOLIDAY','WEEKDAY','WORKINGDAY',
      'WEATHERSIT','TEMP','ATEMP','HUM','WINDSPEED','CASUAL','REGISTERED','CNT']

#Lettura Dataset
ds=pd.read_csv('day.csv', delimiter=",")

log_file_avila = open("01 - InfoFeature/Info feature Dataset_bike_sharing.txt", "w")

for header in headers:

    print(file=log_file_avila)
    print(f"{head[headers.index(header)]}", file=log_file_avila)
    print(ds[header].value_counts().sort_index(), file=log_file_avila)

    #Per ciascun campo andiamo a stampare, infine, il totale dei dati nel dataset al fine di
    #notare eventuali difformità causati da valori assenti
    print(f"Totale: {ds[header].count()}", file=log_file_avila)

ds=ds.drop(['instant','dteday'], axis=1)

ds.to_csv('02 - Elaborated Dataset/Dataset_Bike_Sharing.csv')