#
#AVILA DATA SET
#
#Descrizione del Dataset al seguente link: https://archive.ics.uci.edu/ml/datasets/Avila
#

import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Colonne Dataset

headers=['Intercolumnar distance','upper margin','lower margin','exploitation','row number',
         'modular ratio','interlinear spacing','weight','peak number','modular ratio / interlinear spacing', 'Class']

head=['INTERCOLUMNAR DISTANCE','UPPER MARGIN','LOWER MARGIN','EXPLOITATION','ROW NUMBER',
      'MODULAR RATIO','INTERLINEAR SPACING','WEIGHT','PEAK NUMBER','MODULAR RATIO / INTERLINEAR SPACING', 'CLASS']

#Lettura Dataset
ds1=pd.read_csv('avila-tr.txt', delimiter=",", names=headers)
ds2=pd.read_csv('avila-ts.txt', delimiter=",", names=headers)
ds=pd.concat([ds1, ds2], ignore_index=True)

log_file_avila = open("01 - InfoFeature/Info feature Dataset_avila.txt", "w")

for header in headers:

    print(file=log_file_avila)
    print(f"{head[headers.index(header)]}", file=log_file_avila)
    print(ds[header].value_counts().sort_index(), file=log_file_avila)

    #Per ciascun campo andiamo a stampare, infine, il totale dei dati nel dataset al fine di
    #notare eventuali difformità causati da valori assenti
    print(f"Totale: {ds[header].count()}", file=log_file_avila)

#LABEL-ENCODING
#Per mantenere i dati categorici e poterli quindi utilizzare con gli algoritmi di classificazione prefissati
#c'è bisogno di fare delle lavorazioni al fine di renderli dei dati "numerici" utilizzabili.
#Il Label Encoding identifica per ogni attributo categorico le classi di quell'attributo osservando i valori
#che questo prende ad ogni record; infine assegna ad ogni classe un numero in ordine crescente.
#Un difetto del Label Encoding è che attraverso questa codifica potrebbe indurre in errore il classificatore
#che potrebbe identificare una relazione di precedenza tra le classi (es. [Tokyo, Toronto, Roma] = [0, 1, 2] => Roma>Tokyo)

LEnc=LabelEncoder()
LEAvila=ds
LEAvila['Class'] = LEnc.fit_transform(LEAvila['Class']) 
#LabelEncoder() genera le label per i dati categorici mettendo le stringhe in ordine alfabetico

ds.to_csv('02 - Elaborated Dataset/avila.csv',)