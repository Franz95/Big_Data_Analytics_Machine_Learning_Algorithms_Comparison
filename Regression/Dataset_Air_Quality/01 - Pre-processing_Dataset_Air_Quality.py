#
#AIR QUALITY DATA SET
#
#Descrizione del Dataset al seguente link: https://archive.ics.uci.edu/ml/datasets/Air+Quality
#

import pandas as pd

#Colonne Dataset

headers=['Date','Time','CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)',
         'PT08.S2(NMHC)','NOx(GT)','PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)', 'PT08.S5(O3)','T','RH','AH']

head=['DATE','TIME','CO(GT)','PT08.S1(CO)','NMHC(GT)','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)',
      'PT08.S3(NOx)','NO2(GT)','PT08.S4(NO2)', 'PT08.S5(O3)','T','RH','AH']

#Lettura Dataset
ds=pd.read_csv('AirQualityUCI.csv', delimiter=";")

log_file_avila = open("01 - InfoFeature/Info feature Dataset_air_quality.txt", "w")

for header in headers:

    print(file=log_file_avila)
    print(f"{head[headers.index(header)]}", file=log_file_avila)
    print(ds[header].value_counts().sort_index(), file=log_file_avila)

    #Per ciascun campo andiamo a stampare, infine, il totale dei dati nel dataset al fine di
    #notare eventuali difformità causati da valori assenti
    print(f"Totale: {ds[header].count()}", file=log_file_avila)

dsM=ds.drop(['Date','Time','Unnamed: 15','Unnamed: 16'], axis=1)
dsM['CO(GT)']=dsM['CO(GT)'].replace(",",".",regex=True).astype(float)
dsM['C6H6(GT)']=dsM['C6H6(GT)'].replace(",",".",regex=True).astype(float)
dsM['T']=dsM['T'].replace(",",".",regex=True).astype(float)
dsM['RH']=dsM['RH'].replace(",",".",regex=True).astype(float)
dsM['AH']=dsM['AH'].replace(",",".",regex=True).astype(float)

for i in range(9357, 9471):
    dsM=dsM.drop(i,axis=0)
dsM.to_csv('02 - Elaborated Dataset/Dataset_Air_Quality.csv')