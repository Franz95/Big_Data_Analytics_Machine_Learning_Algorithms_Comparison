#
#BEHAVIOR OF THE URBAN TRAFFIC OF THE CITY OF SAO PAULO IN BRAZIL DATA SET
#
#Descrizione del Dataset al seguente link: https://archive.ics.uci.edu/ml/datasets/Behavior+of+the+urban+traffic+of+the+city+of+Sao+Paulo+in+Brazil
#


import pandas as pd

#Colonne Dataset

headers=['Hour (Coded)','Immobilized bus','Broken Truck','Vehicle excess','Accident victim','Running over',
         'Fire vehicles','Occurrence involving freight','Incident involving dangerous freight','Lack of electricity',
         'Fire', 'Point of flooding','Manifestations','Defect in the network of trolleybuses','Tree on the road',
         'Semaphore off','Intermittent Semaphore','Slowness in traffic (%)']

head=['Hour (Coded)','Immobilized bus','Broken Truck','Vehicle excess','Accident victim','Running over',
      'Fire vehicles','Occurrence involving freight','Incident involving dangerous freight','Lack of electricity',
      'Fire', 'Point of flooding','Manifestations','Defect in the network of trolleybuses','Tree on the road',
      'Semaphore off','Intermittent Semaphore','Slowness in traffic (%)']

#Lettura Dataset
ds=pd.read_csv('Behavior of the urban traffic of the city of Sao Paulo in Brazil.csv', delimiter=";")

log_file_traffic = open("01 - InfoFeature/Info feature Dataset_Traffic_San_Paolo.txt", "w")

for header in headers:

    print(file=log_file_traffic)
    print(f"{head[headers.index(header)]}", file=log_file_traffic)
    print(ds[header].value_counts().sort_index(), file=log_file_traffic)

    #Per ciascun campo andiamo a stampare, infine, il totale dei dati nel dataset al fine di
    #notare eventuali difformità causati da valori assenti
    print(f"Totale: {ds[header].count()}", file=log_file_traffic)

dsM=ds
dsM['Slowness in traffic (%)'] = [int(float(str(i).replace(",","."))) for i in dsM['Slowness in traffic (%)']]
dsM.to_csv('02 - Elaborated Dataset/Dataset_Traffic_San_Paolo.csv')

dsM['Broken Truck']=dsM.pop('Broken Truck')
dsM.to_csv('02 - Elaborated Dataset/Dataset_Traffic_San_Paolo_BT.csv')