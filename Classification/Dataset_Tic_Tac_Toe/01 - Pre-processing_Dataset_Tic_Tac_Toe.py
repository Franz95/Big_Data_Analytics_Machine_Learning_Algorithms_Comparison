#
#TIC-TAC-TOE ENDGAME DATA SET
#
#Descrizione del data set al seguente link: https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
#


import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Colonne Dataset

headers=['Mov_1','Mov_2','Mov_3','Mov_4','Mov_5','Mov_6','Mov_7','Mov_8','Mov_9','Risultato']

head=['MOV_1','MOV_2','MOV_3','MOV_4','MOV_5','MOV_6','MOV_7','MOV_8','MOV_9','RISULTATO']

#Lettura Dataset
ds=pd.read_csv('tic-tac-toe.data.txt', delimiter=",", names=headers)
log_file_Tic_Tac_Toe = open("01 - InfoFeature/Info feature Dataset_Tic_Tac_Toe.txt", "w")

for header in headers:

    print(file=log_file_Tic_Tac_Toe)
    print(f"{head[headers.index(header)]}", file=log_file_Tic_Tac_Toe)
    print(ds[header].value_counts().sort_index(), file=log_file_Tic_Tac_Toe)

    #Per ciascun campo andiamo a stampare, infine, il totale dei dati nel dataset al fine di
    #notare eventuali difformità causati da valori assenti
    print(f"Totale: {ds[header].count()}", file=log_file_Tic_Tac_Toe)

#LABEL-ENCODING
#Per mantenere i dati categorici e poterli quindi utilizzare con gli algoritmi di classificazione prefissati
#c'è bisogno di fare delle lavorazioni al fine di renderli dei dati "numerici" utilizzabili.
#Il Label Encoding identifica per ogni attributo categorico le classi di quell'attributo osservando i valori
#che questo prende ad ogni record; infine assegna ad ogni classe un numero in ordine crescente.
#Un difetto del Label Encoding è che attraverso questa codifica potrebbe indurre in errore il classificatore
#che potrebbe identificare una relazione di precedenza tra le classi (es. [Tokyo, Toronto, Roma] = [0, 1, 2] => Roma>Tokyo)

LEnc=LabelEncoder()

LETTT=ds
for att in headers:
    LETTT[att] = LEnc.fit_transform(LETTT[att]) 
#LabelEncoder() genera le label per i dati categorici mettendo le stringhe in ordine alfabetico da questo sappiamo per certo che il dataset in
#esame avrà tutte le colonne che lo compongono formate da [0,1,2] dove 0=b, 1=o, 2=x
LETTT.to_csv('02 - Elaborated Dataset/tic-tac-toe.csv',)