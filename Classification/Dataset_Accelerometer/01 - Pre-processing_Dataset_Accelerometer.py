#
#ACCELEROMETER DATA SET
#
#Descrizione del data set al seguente link: https://archive.ics.uci.edu/ml/datasets/Accelerometer
#

import pandas as pd

#Colonne Dataset

headers=['wconfid','pctid','x','y','z']

head=['WCONFID','PCTID','X','Y','Z']

#Lettura Dataset
ds=pd.read_csv('accelerometer.csv', delimiter=",")
log_file_accelerometer = open("01 - InfoFeature/Info feature Dataset_Accelerometer.txt", "w")

for header in headers:

    print(file=log_file_accelerometer)
    print(f"{head[headers.index(header)]}", file=log_file_accelerometer)
    print(ds[header].value_counts().sort_index(), file=log_file_accelerometer)

    #Per ciascun campo andiamo a stampare, infine, il totale dei dati nel dataset al fine di
    #notare eventuali difformità causati da valori assenti
    print(f"Totale: {ds[header].count()}", file=log_file_accelerometer)

ds['wconfid'] = ds.pop('wconfid')
ds.to_csv('02 - Elaborated Dataset/accelerometer.csv',)