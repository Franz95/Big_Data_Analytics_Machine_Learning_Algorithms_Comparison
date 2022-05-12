#
#Wireless Indoor Localization
#
#Descrizione del Dataset al seguente link: https://archive.ics.uci.edu/ml/datasets/Wireless+Indoor+Localization
#


import pandas as pd

#Colonne Dataset

headers=['Sig_1','Sig_2','Sig_3','Sig_4','Sig_5','Sig_6','Sig_7','Stanza']

head=['SIG_1','SIG_2','SIG_3','SIG_4','SIG_5','SIG_6','SIG_7','STANZA']

#Lettura Dataset
ds=pd.read_csv('wifi_localization.txt', delim_whitespace=True, names=headers)

log_file_wifi = open("01 - InfoFeature/Info feature Dataset_wifi.txt", "w")

for header in headers:

    print(file=log_file_wifi)
    print(f"{head[headers.index(header)]}", file=log_file_wifi)
    print(ds[header].value_counts().sort_index(), file=log_file_wifi)

    #Per ciascun campo andiamo a stampare, infine, il totale dei dati nel dataset al fine di
    #notare eventuali difformità causati da valori assenti
    print(f"Totale: {ds[header].count()}", file=log_file_wifi)

ds.to_csv('02 - Elaborated Dataset/wifi_localization.csv')