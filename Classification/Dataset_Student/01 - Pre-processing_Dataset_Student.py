#
#STUDENT PERFORMANCE DATA SET
#
#Descrizione del data set al seguente link: https://archive.ics.uci.edu/ml/datasets/Student+Performance
#

from sklearn.preprocessing import LabelEncoder
import pandas as pd

#Colonne Dataset
headers=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason',
         'guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities',
         'nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc',
         'health','absences','G1','G2','G3']

head=['SCHOOL','SEX','AGE','ADDRESS','FAMSIZE','PSTATUS','MEDU','FEDU','MJOB','FJOB',
      'REASON','GUARDIAN','TRAVELTIME','STUDYTIME','FAILURES','SCHOOLSUP','FAMSUP',
      'PAID','ACTIVITIES','NURSERY','HIGHER','INTERNET','ROMANTIC','FAMREL','FREETIME',
      'GOOUT','DALC','WALC','HEALTH','ABSENCES','G1','G2','G3']

#Lettura Dataset
dsMath=pd.read_csv('student-mat.csv', delimiter=";") #carichiamo database con votazioni Matematica
dsPort=pd.read_csv('student-por.csv', delimiter=";") #carichiamo database con votazioni Portoghese

#ANALISI DEI CAMPI DEL DATASET
#Al fine di poter effettuare lavorazioni sui dati abbiamo prima bisogno di sapere per ciascuna feauture quali valori assume
#e le quantità di dati corrispondenti ad ogni valore; per fare ciò creiamo un file log in cui andiamo a scrivere tutte queste info.

log_file_Math = open("01 - InfoFeature/Info feature Dataset_Student_Math.txt", "w") #apriamo il file in sola scrittura
log_file_Port = open("01 - InfoFeature/Info feature Dataset_Student_Port.txt", "w") #apriamo il file in sola scrittura

print("DATASET MATEMATICA", file=log_file_Math)
print("DATASET PORTOGHESE", file=log_file_Port)

for header in headers:

    #STAMPA DATASET MATEMATICA
    print(file=log_file_Math)
    print(f"{head[headers.index(header)]}", file=log_file_Math)
    
    #In questo blocco if-else andiamo a stampare in ordine decrescente di indice
    #Medu e Fedu in quanto indicano il grado di istruzione dei genitori del singolo studente
    #e l'indice 4 è quello assegnato al livello di istruzione maggiore.
    #Andiamo invece a stampare Mjob, Fjob, reason e guardian in ordine decrescente di valore in modo
    #da evidenziare nel file i valori maggioritari

    if(header=='Medu' or header=='Fedu'):
        print(dsMath[header].value_counts().sort_index(ascending=False), file=log_file_Math)
    elif(header=='Mjob' or header=='Fjob' or header=='reason' or header=='guardian'):
        print(dsMath[header].value_counts().sort_values(ascending=False), file=log_file_Math)
    else:
        print(dsMath[header].value_counts().sort_index(), file=log_file_Math)

    #Per ciascun campo andiamo a stampare, infine, il totale dei dati nel dataset al fine di
    #notare eventuali difformità causati da valori assenti
    print(f"Totale: {dsMath[header].count()}", file=log_file_Math)

    #STAMPA DATASET PORTOGHESE
    print(file=log_file_Port)
    print(f"{head[headers.index(header)]}", file=log_file_Port)
    
    if(header=='Medu' or header=='Fedu'):
        print(dsPort[header].value_counts().sort_index(ascending=False), file=log_file_Port)
    elif(header=='Mjob' or header=='Fjob' or header=='reason' or header=='guardian'):
        print(dsPort[header].value_counts().sort_values(ascending=False), file=log_file_Port)
    else:
        print(dsPort[header].value_counts().sort_index(), file=log_file_Port)

    print(f"Totale: {dsPort[header].count()}", file=log_file_Port)

#PREPARAZIONE DATASETS

#ELIMINAZIONE DATI CATEGORICI
#Come detto nell'introduzione non processa dati categorici quindi andiamo ad eliminare tutte le colonne che contengono questo tipo di dati.
NCMath= dsMath.drop(['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian',
                     'schoolsup','famsup','paid','activities','nursery','higher','internet',
                     'romantic'], axis=1)
NCPort= dsPort.drop(['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian',
                     'schoolsup','famsup','paid','activities','nursery','higher','internet',
                     'romantic'], axis=1)

NCMath['age']=NCMath.pop('age')
NCPort['age']=NCPort.pop('age')

#LABEL-ENCODING
#Per mantenere i dati categorici e poterli quindi utilizzare con gli algoritmi di classificazione prefissati
#c'è bisogno di fare delle lavorazioni al fine di renderli dei dati "numerici" utilizzabili.
#Il Label Encoding identifica per ogni attributo categorico le classi di quell'attributo osservando i valori
#che questo prende ad ogni record; infine assegna ad ogni classe un numero in ordine crescente.
#Un difetto del Label Encoding è che attraverso questa codifica potrebbe indurre in errore il classificatore
#che potrebbe identificare una relazione di precedenza tra le classi (es. [Tokyo, Toronto, Roma] = [0, 1, 2] => Roma>Tokyo)

LEnc=LabelEncoder()
cat_attribs = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
               'schoolsup','famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

LEMath=dsMath
LEPort=dsPort
for att in cat_attribs:
    LEMath[att] = LEnc.fit_transform(LEMath[att])
    LEPort[att] = LEnc.fit_transform(LEPort[att])

LEMath['age']=LEMath.pop('age')
LEPort['age']=LEPort.pop('age')

#Salviamo i dataset preprocessati nelle rispettive cartelle per permettere 
#a ogni script di poter fare le future elaborazioni
NCMath.to_csv('02 - Elaborated Dataset/02.1 - NoEncoding/Dataset_Student_Math_NE.csv')
NCPort.to_csv('02 - Elaborated Dataset/02.1 - NoEncoding/Dataset_Student_Port_NE.csv')
LEMath.to_csv('02 - Elaborated Dataset/02.2 - LabelEncoding/Dataset_Student_Math_LE.csv')
LEPort.to_csv('02 - Elaborated Dataset/02.2 - LabelEncoding/Dataset_Student_Port_LE.csv')