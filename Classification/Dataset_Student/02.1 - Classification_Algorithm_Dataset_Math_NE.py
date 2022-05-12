from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from timeit import default_timer as timer
import xgboost as xgb
import pandas as pd

data= pd.read_csv("02 - Elaborated Dataset/02.1 - NoEncoding/Dataset_Student_Math_NE.csv")

#Dichiariamo i modelli che utilizzeremo
modelXGB=xgb.XGBClassifier()
modelSVC=SVC()
modelDTC=DecisionTreeClassifier()

#Dividiamo il dataset selezionando in X tutte le feature che utilizzeremo per addestrare il modello
#in Y la colonna che utilizzeremo per controllare le performance del test.
X, y = data.iloc[:,:-1], data.iloc[:,-1]
data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)

start=timer()
modelXGB.fit(X_train, y_train)
end=timer()
XGB_FITTING_ELAPSED_TIME=round(end - start,2)*1000

start=timer()
modelSVC.fit(X_train, y_train)
end=timer()
SVC_FITTING_ELAPSED_TIME=round(end - start,2)*1000

start=timer()
modelDTC.fit(X_train, y_train)
end=timer()
DTC_FITTING_ELAPSED_TIME = round(end - start,2)*1000

start=timer()
y_predsXGB = modelXGB.predict(X_test)
predictionsXGB=[round(value) for value in y_predsXGB]
accuracyXGB=accuracy_score(y_test,predictionsXGB)
recallXGB=recall_score(y_test,predictionsXGB,average='weighted')
precisionXGB=precision_score(y_test,predictionsXGB,average='weighted')
f1XGB=f1_score(y_test,predictionsXGB,average='weighted')
end=timer()
XGB_EVALUATIONS_ELAPSED_TIME = round(end - start,2)*1000

start=timer()
y_predsDT = modelDTC.predict(X_test)
predictionsDTC=[round(value) for value in y_predsDT]
accuracyDTC=accuracy_score(y_test,predictionsDTC)
recallDTC=recall_score(y_test,predictionsDTC,average='weighted')
precisionDTC=precision_score(y_test,predictionsDTC,average='weighted')
f1DTC=f1_score(y_test,predictionsDTC,average='weighted')
end=timer()
DTC_EVALUATIONS_ELAPSED_TIME = round(end - start,2)*1000

start=timer()
y_predsSVC = modelSVC.predict(X_test)
predictionsSVC=[round(value) for value in y_predsSVC]
accuracySVC=accuracy_score(y_test,predictionsSVC)
recallSVC=recall_score(y_test,predictionsSVC,average='weighted')
precisionSVC=precision_score(y_test,predictionsSVC,average='weighted')
f1SVC=f1_score(y_test,predictionsSVC,average='weighted')
end=timer()
SVC_EVALUATIONS_ELAPSED_TIME = round(end - start,2)*1000

#CROSS VALIDATION 3-SPLITS
kfold=KFold(n_splits=3, shuffle=True, random_state=9)
scoring=['accuracy','precision_micro', 'recall_micro', 'f1_micro']

start=timer()
CV_XGB_KFOLD3_ACCURACY= cross_val_score(modelXGB,X,y, cv=kfold, scoring='accuracy')
end=timer()
CV_XGB_KFOLD3_ACCURACY_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD3_ACCURACY= cross_val_score(modelDTC,X,y, cv=kfold, scoring='accuracy')
end=timer()
CV_DTC_KFOLD3_ACCURACY_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD3_ACCURACY= cross_val_score(modelSVC,X,y, cv=kfold, scoring='accuracy')
end=timer()
CV_SVC_KFOLD3_ACCURACY_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_XGB_KFOLD3_PRECISION= cross_val_score(modelXGB,X,y,cv=kfold,scoring='precision_micro')
end=timer()
CV_XGB_KFOLD3_PRECISION_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD3_PRECISION= cross_val_score(modelDTC,X,y,cv=kfold,scoring='precision_micro')
end=timer()
CV_DTC_KFOLD3_PRECISION_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD3_PRECISION= cross_val_score(modelSVC,X,y,cv=kfold,scoring='precision_micro')
end=timer()
CV_SVC_KFOLD3_PRECISION_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_XGB_KFOLD3_RECALL= cross_val_score(modelXGB,X,y,cv=kfold,scoring='recall_micro')
end=timer()
CV_XGB_KFOLD3_RECALL_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD3_RECALL= cross_val_score(modelDTC,X,y,cv=kfold,scoring='recall_micro')
end=timer()
CV_DTC_KFOLD3_RECALL_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD3_RECALL= cross_val_score(modelSVC,X,y,cv=kfold,scoring='recall_micro')
end=timer()
CV_SVC_KFOLD3_RECALL_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_XGB_KFOLD3_F1= cross_val_score(modelXGB,X,y,cv=kfold,scoring='f1_micro')
end=timer()
CV_XGB_KFOLD3_F1_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD3_F1= cross_val_score(modelDTC,X,y,cv=kfold,scoring='f1_micro')
end=timer()
CV_DTC_KFOLD3_F1_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD3_F1= cross_val_score(modelSVC,X,y,cv=kfold,scoring='f1_micro')
end=timer()
CV_SVC_KFOLD3_F1_ELAPSED_TIME= round(end - start, 2)*1000

#CROSS VALIDATION 5-SPLITS
kfold=KFold(n_splits=5, shuffle=True, random_state=9)
scoring=['accuracy','precision_micro', 'recall_micro', 'f1_micro']

start=timer()
CV_XGB_KFOLD5_ACCURACY= cross_val_score(modelXGB,X,y, cv=kfold, scoring='accuracy')
end=timer()
CV_XGB_KFOLD5_ACCURACY_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD5_ACCURACY= cross_val_score(modelDTC,X,y, cv=kfold, scoring='accuracy')
end=timer()
CV_DTC_KFOLD5_ACCURACY_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD5_ACCURACY= cross_val_score(modelSVC,X,y, cv=kfold, scoring='accuracy')
end=timer()
CV_SVC_KFOLD5_ACCURACY_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_XGB_KFOLD5_PRECISION= cross_val_score(modelXGB,X,y,cv=kfold,scoring='precision_micro')
end=timer()
CV_XGB_KFOLD5_PRECISION_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD5_PRECISION= cross_val_score(modelDTC,X,y,cv=kfold,scoring='precision_micro')
end=timer()
CV_DTC_KFOLD5_PRECISION_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD5_PRECISION= cross_val_score(modelSVC,X,y,cv=kfold,scoring='precision_micro')
end=timer()
CV_SVC_KFOLD5_PRECISION_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_XGB_KFOLD5_RECALL= cross_val_score(modelXGB,X,y,cv=kfold,scoring='recall_micro')
end=timer()
CV_XGB_KFOLD5_RECALL_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD5_RECALL= cross_val_score(modelDTC,X,y,cv=kfold,scoring='recall_micro')
end=timer()
CV_DTC_KFOLD5_RECALL_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD5_RECALL= cross_val_score(modelSVC,X,y,cv=kfold,scoring='recall_micro')
end=timer()
CV_SVC_KFOLD5_RECALL_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_XGB_KFOLD5_F1= cross_val_score(modelXGB,X,y,cv=kfold,scoring='f1_micro')
end=timer()
CV_XGB_KFOLD5_F1_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD5_F1= cross_val_score(modelDTC,X,y,cv=kfold,scoring='f1_micro')
end=timer()
CV_DTC_KFOLD5_F1_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD5_F1= cross_val_score(modelSVC,X,y,cv=kfold,scoring='f1_micro')
end=timer()
CV_SVC_KFOLD5_F1_ELAPSED_TIME= round(end - start, 2)*1000

#CROSS VALIDATION 10-SPLITS
kfold=KFold(n_splits=10, shuffle=True, random_state=9)
scoring=['accuracy','precision_micro', 'recall_micro', 'f1_micro']

start=timer()
CV_XGB_KFOLD10_ACCURACY= cross_val_score(modelXGB,X,y, cv=kfold, scoring='accuracy')
end=timer()
CV_XGB_KFOLD10_ACCURACY_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD10_ACCURACY= cross_val_score(modelDTC,X,y, cv=kfold, scoring='accuracy')
end=timer()
CV_DTC_KFOLD10_ACCURACY_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD10_ACCURACY= cross_val_score(modelSVC,X,y, cv=kfold, scoring='accuracy')
end=timer()
CV_SVC_KFOLD10_ACCURACY_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_XGB_KFOLD10_PRECISION= cross_val_score(modelXGB,X,y,cv=kfold,scoring='precision_micro')
end=timer()
CV_XGB_KFOLD10_PRECISION_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD10_PRECISION= cross_val_score(modelDTC,X,y,cv=kfold,scoring='precision_micro')
end=timer()
CV_DTC_KFOLD10_PRECISION_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD10_PRECISION= cross_val_score(modelSVC,X,y,cv=kfold,scoring='precision_micro')
end=timer()
CV_SVC_KFOLD10_PRECISION_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_XGB_KFOLD10_RECALL= cross_val_score(modelXGB,X,y,cv=kfold,scoring='recall_micro')
end=timer()
CV_XGB_KFOLD10_RECALL_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD10_RECALL= cross_val_score(modelDTC,X,y,cv=kfold,scoring='recall_micro')
end=timer()
CV_DTC_KFOLD10_RECALL_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD10_RECALL= cross_val_score(modelSVC,X,y,cv=kfold,scoring='recall_micro')
end=timer()
CV_SVC_KFOLD10_RECALL_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_XGB_KFOLD10_F1= cross_val_score(modelXGB,X,y,cv=kfold,scoring='f1_micro')
end=timer()
CV_XGB_KFOLD10_F1_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTC_KFOLD10_F1= cross_val_score(modelDTC,X,y,cv=kfold,scoring='f1_micro')
end=timer()
CV_DTC_KFOLD10_F1_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVC_KFOLD10_F1= cross_val_score(modelSVC,X,y,cv=kfold,scoring='f1_micro')
end=timer()
CV_SVC_KFOLD10_F1_ELAPSED_TIME= round(end - start, 2)*1000

#GENERIAMO I CSV E GLI EXCEL CON TUTTI I RISULTATI RACCOLTI DALLE ANALISI EFFETTUATE
indices=['XGB','DTC','SVC']
df = pd.DataFrame(columns=['Accuracy','Precision','Recall','F1','Fit_Elapsed_Time','Evaluation_Elapsed_Time'], index=indices)
                           
df.loc['XGB']=pd.Series({'Accuracy': "%.2f%%" % (accuracyXGB*100),
                         'Precision': "%.2f%%" % (precisionXGB*100),
                         'Recall': "%.2f%%" % (recallXGB*100),
                         'F1': "%.2f%%" % (f1XGB*100),
                         'Fit_Elapsed_Time':"%.2f ms" % XGB_FITTING_ELAPSED_TIME,
                         'Evaluation_Elapsed_Time':"%.2f ms" % XGB_EVALUATIONS_ELAPSED_TIME
                         })

df.loc['DTC']=pd.Series({'Accuracy': "%.2f%%" % (accuracyDTC*100),
                         'Precision': "%.2f%%" % (precisionDTC*100),
                         'Recall': "%.2f%%" % (recallDTC*100),
                         'F1': "%.2f%%" % (f1DTC*100),
                         'Fit_Elapsed_Time':"%.2f ms" % DTC_FITTING_ELAPSED_TIME,
                         'Evaluation_Elapsed_Time':"%.2f ms" % DTC_EVALUATIONS_ELAPSED_TIME
                         })

df.loc['SVC']=pd.Series({'Accuracy': "%.2f%%" % (accuracySVC*100),
                         'Precision': "%.2f%%" % (precisionSVC*100),
                         'Recall': "%.2f%%" % (recallSVC*100),
                         'F1': "%.2f%%" % (f1SVC*100),
                         'Fit_Elapsed_Time':"%.2f ms" % SVC_FITTING_ELAPSED_TIME,
                         'Evaluation_Elapsed_Time':"%.2f ms" % SVC_EVALUATIONS_ELAPSED_TIME
                         })

dfcvKF3 = pd.DataFrame(columns=['Kfold_Max_Accuracy','Kfold_Mean_Accuracy','Kfold_Min_Accuracy','Kfold_Std_Deviation_Accuracy','Kfold_Accuracy_Elapsed_Time',
                           'Kfold_Max_Precision','Kfold_Mean_Precision','Kfold_Min_Precision','Kfold_Std_Deviation_Precision','Kfold_Precision_Elapsed_Time',
                           'Kfold_Max_Recall','Kfold_Mean_Recall','Kfold_Min_Recall','Kfold_Std_Deviation_Recall','Kfold_Recall_Elapsed_Time',
                           'Kfold_Max_F1','Kfold_Mean_F1','Kfold_Min_F1','Kfold_Std_Deviation_F1','Kfold_F1_Elapsed_Time'],index=indices)
                           
dfcvKF3.loc['XGB']=pd.Series({'Kfold_Max_Accuracy': "%.2f%%" % (CV_XGB_KFOLD3_ACCURACY.max()*100),
                         'Kfold_Mean_Accuracy': "%.2f%%" % (CV_XGB_KFOLD3_ACCURACY.mean()*100),
                         'Kfold_Min_Accuracy': "%.2f%%" % (CV_XGB_KFOLD3_ACCURACY.min()*100),
                         'Kfold_Std_Deviation_Accuracy': "%.2f%%" % (CV_XGB_KFOLD3_ACCURACY.std()*100),
                         'Kfold_Accuracy_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD3_ACCURACY_ELAPSED_TIME,
                         'Kfold_Max_Precision': "%.2f%%" % (CV_XGB_KFOLD3_PRECISION.max()*100),
                         'Kfold_Mean_Precision': "%.2f%%" % (CV_XGB_KFOLD3_PRECISION.mean()*100),
                         'Kfold_Min_Precision': "%.2f%%" % (CV_XGB_KFOLD3_PRECISION.min()*100),
                         'Kfold_Std_Deviation_Precision': "%.2f%%" % (CV_XGB_KFOLD3_PRECISION.std()*100),
                         'Kfold_Precision_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD3_PRECISION_ELAPSED_TIME,
                         'Kfold_Max_Recall': "%.2f%%" % (CV_XGB_KFOLD3_RECALL.max()*100),
                         'Kfold_Mean_Recall': "%.2f%%" % (CV_XGB_KFOLD3_RECALL.mean()*100),
                         'Kfold_Min_Recall': "%.2f%%" % (CV_XGB_KFOLD3_RECALL.min()*100),
                         'Kfold_Std_Deviation_Recall': "%.2f%%" % (CV_XGB_KFOLD3_RECALL.std()*100),
                         'Kfold_Recall_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD3_RECALL_ELAPSED_TIME,
                         'Kfold_Max_F1': "%.2f%%" % (CV_XGB_KFOLD3_F1.max()*100),
                         'Kfold_Mean_F1': "%.2f%%" % (CV_XGB_KFOLD3_F1.mean()*100),
                         'Kfold_Min_F1': "%.2f%%" % (CV_XGB_KFOLD3_F1.min()*100),
                         'Kfold_Std_Deviation_F1': "%.2f%%" % (CV_XGB_KFOLD3_F1.std()*100),
                         'Kfold_F1_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD3_F1_ELAPSED_TIME
                         })

dfcvKF3.loc['DTC']=pd.Series({'Kfold_Max_Accuracy': "%.2f%%" % (CV_DTC_KFOLD3_ACCURACY.max()*100),
                         'Kfold_Mean_Accuracy': "%.2f%%" % (CV_DTC_KFOLD3_ACCURACY.mean()*100),
                         'Kfold_Min_Accuracy': "%.2f%%" % (CV_DTC_KFOLD3_ACCURACY.min()*100),
                         'Kfold_Std_Deviation_Accuracy': "%.2f%%" % (CV_DTC_KFOLD3_ACCURACY.std()*100),
                         'Kfold_Accuracy_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD3_ACCURACY_ELAPSED_TIME,
                         'Kfold_Max_Precision': "%.2f%%" % (CV_DTC_KFOLD3_PRECISION.max()*100),
                         'Kfold_Mean_Precision': "%.2f%%" % (CV_DTC_KFOLD3_PRECISION.mean()*100),
                         'Kfold_Min_Precision': "%.2f%%" % (CV_DTC_KFOLD3_PRECISION.min()*100),
                         'Kfold_Std_Deviation_Precision': "%.2f%%" % (CV_DTC_KFOLD3_PRECISION.std()*100),
                         'Kfold_Precision_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD3_PRECISION_ELAPSED_TIME,
                         'Kfold_Max_Recall': "%.2f%%" % (CV_DTC_KFOLD3_RECALL.max()*100),
                         'Kfold_Mean_Recall': "%.2f%%" % (CV_DTC_KFOLD3_RECALL.mean()*100),
                         'Kfold_Min_Recall': "%.2f%%" % (CV_DTC_KFOLD3_RECALL.min()*100),
                         'Kfold_Std_Deviation_Recall': "%.2f%%" % (CV_DTC_KFOLD3_RECALL.std()*100),
                         'Kfold_Recall_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD3_RECALL_ELAPSED_TIME,
                         'Kfold_Max_F1': "%.2f%%" % (CV_DTC_KFOLD3_F1.max()*100),
                         'Kfold_Mean_F1': "%.2f%%" % (CV_DTC_KFOLD3_F1.mean()*100),
                         'Kfold_Min_F1': "%.2f%%" % (CV_DTC_KFOLD3_F1.min()*100),
                         'Kfold_Std_Deviation_F1': "%.2f%%" % (CV_DTC_KFOLD3_F1.std()*100),
                         'Kfold_F1_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD3_F1_ELAPSED_TIME
                         })

dfcvKF3.loc['SVC']=pd.Series({'Kfold_Max_Accuracy': "%.2f%%" % (CV_SVC_KFOLD3_ACCURACY.max()*100),
                         'Kfold_Mean_Accuracy': "%.2f%%" % (CV_SVC_KFOLD3_ACCURACY.mean()*100),
                         'Kfold_Min_Accuracy': "%.2f%%" % (CV_SVC_KFOLD3_ACCURACY.min()*100),
                         'Kfold_Std_Deviation_Accuracy': "%.2f%%" % (CV_SVC_KFOLD3_ACCURACY.std()*100),
                         'Kfold_Accuracy_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD3_ACCURACY_ELAPSED_TIME,
                         'Kfold_Max_Precision': "%.2f%%" % (CV_SVC_KFOLD3_PRECISION.max()*100),
                         'Kfold_Mean_Precision': "%.2f%%" % (CV_SVC_KFOLD3_PRECISION.mean()*100),
                         'Kfold_Min_Precision': "%.2f%%" % (CV_SVC_KFOLD3_PRECISION.min()*100),
                         'Kfold_Std_Deviation_Precision': "%.2f%%" % (CV_SVC_KFOLD3_PRECISION.std()*100),
                         'Kfold_Precision_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD3_PRECISION_ELAPSED_TIME,
                         'Kfold_Max_Recall': "%.2f%%" % (CV_SVC_KFOLD3_RECALL.max()*100),
                         'Kfold_Mean_Recall': "%.2f%%" % (CV_SVC_KFOLD3_RECALL.mean()*100),
                         'Kfold_Min_Recall': "%.2f%%" % (CV_SVC_KFOLD3_RECALL.min()*100),
                         'Kfold_Std_Deviation_Recall': "%.2f%%" % (CV_SVC_KFOLD3_RECALL.std()*100),
                         'Kfold_Recall_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD3_RECALL_ELAPSED_TIME,
                         'Kfold_Max_F1': "%.2f%%" % (CV_SVC_KFOLD3_F1.max()*100),
                         'Kfold_Mean_F1': "%.2f%%" % (CV_SVC_KFOLD3_F1.mean()*100),
                         'Kfold_Min_F1': "%.2f%%" % (CV_SVC_KFOLD3_F1.min()*100),
                         'Kfold_Std_Deviation_F1': "%.2f%%" % (CV_SVC_KFOLD3_F1.std()*100),
                         'Kfold_F1_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD3_F1_ELAPSED_TIME
                         })

dfcvKF5 = pd.DataFrame(columns=['Kfold_Max_Accuracy','Kfold_Mean_Accuracy','Kfold_Min_Accuracy','Kfold_Std_Deviation_Accuracy','Kfold_Accuracy_Elapsed_Time',
                           'Kfold_Max_Precision','Kfold_Mean_Precision','Kfold_Min_Precision','Kfold_Std_Deviation_Precision','Kfold_Precision_Elapsed_Time',
                           'Kfold_Max_Recall','Kfold_Mean_Recall','Kfold_Min_Recall','Kfold_Std_Deviation_Recall','Kfold_Recall_Elapsed_Time',
                           'Kfold_Max_F1','Kfold_Mean_F1','Kfold_Min_F1','Kfold_Std_Deviation_F1','Kfold_F1_Elapsed_Time'],index=indices)
                           
dfcvKF5.loc['XGB']=pd.Series({'Kfold_Max_Accuracy': "%.2f%%" % (CV_XGB_KFOLD5_ACCURACY.max()*100),
                         'Kfold_Mean_Accuracy': "%.2f%%" % (CV_XGB_KFOLD5_ACCURACY.mean()*100),
                         'Kfold_Min_Accuracy': "%.2f%%" % (CV_XGB_KFOLD5_ACCURACY.min()*100),
                         'Kfold_Std_Deviation_Accuracy': "%.2f%%" % (CV_XGB_KFOLD5_ACCURACY.std()*100),
                         'Kfold_Accuracy_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD5_ACCURACY_ELAPSED_TIME,
                         'Kfold_Max_Precision': "%.2f%%" % (CV_XGB_KFOLD5_PRECISION.max()*100),
                         'Kfold_Mean_Precision': "%.2f%%" % (CV_XGB_KFOLD5_PRECISION.mean()*100),
                         'Kfold_Min_Precision': "%.2f%%" % (CV_XGB_KFOLD5_PRECISION.min()*100),
                         'Kfold_Std_Deviation_Precision': "%.2f%%" % (CV_XGB_KFOLD5_PRECISION.std()*100),
                         'Kfold_Precision_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD5_PRECISION_ELAPSED_TIME,
                         'Kfold_Max_Recall': "%.2f%%" % (CV_XGB_KFOLD5_RECALL.max()*100),
                         'Kfold_Mean_Recall': "%.2f%%" % (CV_XGB_KFOLD5_RECALL.mean()*100),
                         'Kfold_Min_Recall': "%.2f%%" % (CV_XGB_KFOLD5_RECALL.min()*100),
                         'Kfold_Std_Deviation_Recall': "%.2f%%" % (CV_XGB_KFOLD5_RECALL.std()*100),
                         'Kfold_Recall_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD5_RECALL_ELAPSED_TIME,
                         'Kfold_Max_F1': "%.2f%%" % (CV_XGB_KFOLD5_F1.max()*100),
                         'Kfold_Mean_F1': "%.2f%%" % (CV_XGB_KFOLD5_F1.mean()*100),
                         'Kfold_Min_F1': "%.2f%%" % (CV_XGB_KFOLD5_F1.min()*100),
                         'Kfold_Std_Deviation_F1': "%.2f%%" % (CV_XGB_KFOLD5_F1.std()*100),
                         'Kfold_F1_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD5_F1_ELAPSED_TIME
                         })

dfcvKF5.loc['DTC']=pd.Series({'Kfold_Max_Accuracy': "%.2f%%" % (CV_DTC_KFOLD5_ACCURACY.max()*100),
                         'Kfold_Mean_Accuracy': "%.2f%%" % (CV_DTC_KFOLD5_ACCURACY.mean()*100),
                         'Kfold_Min_Accuracy': "%.2f%%" % (CV_DTC_KFOLD5_ACCURACY.min()*100),
                         'Kfold_Std_Deviation_Accuracy': "%.2f%%" % (CV_DTC_KFOLD5_ACCURACY.std()*100),
                         'Kfold_Accuracy_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD5_ACCURACY_ELAPSED_TIME,
                         'Kfold_Max_Precision': "%.2f%%" % (CV_DTC_KFOLD5_PRECISION.max()*100),
                         'Kfold_Mean_Precision': "%.2f%%" % (CV_DTC_KFOLD5_PRECISION.mean()*100),
                         'Kfold_Min_Precision': "%.2f%%" % (CV_DTC_KFOLD5_PRECISION.min()*100),
                         'Kfold_Std_Deviation_Precision': "%.2f%%" % (CV_DTC_KFOLD5_PRECISION.std()*100),
                         'Kfold_Precision_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD5_PRECISION_ELAPSED_TIME,
                         'Kfold_Max_Recall': "%.2f%%" % (CV_DTC_KFOLD5_RECALL.max()*100),
                         'Kfold_Mean_Recall': "%.2f%%" % (CV_DTC_KFOLD5_RECALL.mean()*100),
                         'Kfold_Min_Recall': "%.2f%%" % (CV_DTC_KFOLD5_RECALL.min()*100),
                         'Kfold_Std_Deviation_Recall': "%.2f%%" % (CV_DTC_KFOLD5_RECALL.std()*100),
                         'Kfold_Recall_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD5_RECALL_ELAPSED_TIME,
                         'Kfold_Max_F1': "%.2f%%" % (CV_DTC_KFOLD5_F1.max()*100),
                         'Kfold_Mean_F1': "%.2f%%" % (CV_DTC_KFOLD5_F1.mean()*100),
                         'Kfold_Min_F1': "%.2f%%" % (CV_DTC_KFOLD5_F1.min()*100),
                         'Kfold_Std_Deviation_F1': "%.2f%%" % (CV_DTC_KFOLD5_F1.std()*100),
                         'Kfold_F1_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD5_F1_ELAPSED_TIME
                         })

dfcvKF5.loc['SVC']=pd.Series({'Kfold_Max_Accuracy': "%.2f%%" % (CV_SVC_KFOLD5_ACCURACY.max()*100),
                         'Kfold_Mean_Accuracy': "%.2f%%" % (CV_SVC_KFOLD5_ACCURACY.mean()*100),
                         'Kfold_Min_Accuracy': "%.2f%%" % (CV_SVC_KFOLD5_ACCURACY.min()*100),
                         'Kfold_Std_Deviation_Accuracy': "%.2f%%" % (CV_SVC_KFOLD5_ACCURACY.std()*100),
                         'Kfold_Accuracy_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD5_ACCURACY_ELAPSED_TIME,
                         'Kfold_Max_Precision': "%.2f%%" % (CV_SVC_KFOLD5_PRECISION.max()*100),
                         'Kfold_Mean_Precision': "%.2f%%" % (CV_SVC_KFOLD5_PRECISION.mean()*100),
                         'Kfold_Min_Precision': "%.2f%%" % (CV_SVC_KFOLD5_PRECISION.min()*100),
                         'Kfold_Std_Deviation_Precision': "%.2f%%" % (CV_SVC_KFOLD5_PRECISION.std()*100),
                         'Kfold_Precision_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD5_PRECISION_ELAPSED_TIME,
                         'Kfold_Max_Recall': "%.2f%%" % (CV_SVC_KFOLD5_RECALL.max()*100),
                         'Kfold_Mean_Recall': "%.2f%%" % (CV_SVC_KFOLD5_RECALL.mean()*100),
                         'Kfold_Min_Recall': "%.2f%%" % (CV_SVC_KFOLD5_RECALL.min()*100),
                         'Kfold_Std_Deviation_Recall': "%.2f%%" % (CV_SVC_KFOLD5_RECALL.std()*100),
                         'Kfold_Recall_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD5_RECALL_ELAPSED_TIME,
                         'Kfold_Max_F1': "%.2f%%" % (CV_SVC_KFOLD5_F1.max()*100),
                         'Kfold_Mean_F1': "%.2f%%" % (CV_SVC_KFOLD5_F1.mean()*100),
                         'Kfold_Min_F1': "%.2f%%" % (CV_SVC_KFOLD5_F1.min()*100),
                         'Kfold_Std_Deviation_F1': "%.2f%%" % (CV_SVC_KFOLD5_F1.std()*100),
                         'Kfold_F1_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD5_F1_ELAPSED_TIME
                         })

dfcvKF10 = pd.DataFrame(columns=['Kfold_Max_Accuracy','Kfold_Mean_Accuracy','Kfold_Min_Accuracy','Kfold_Std_Deviation_Accuracy','Kfold_Accuracy_Elapsed_Time',
                           'Kfold_Max_Precision','Kfold_Mean_Precision','Kfold_Min_Precision','Kfold_Std_Deviation_Precision','Kfold_Precision_Elapsed_Time',
                           'Kfold_Max_Recall','Kfold_Mean_Recall','Kfold_Min_Recall','Kfold_Std_Deviation_Recall','Kfold_Recall_Elapsed_Time',
                           'Kfold_Max_F1','Kfold_Mean_F1','Kfold_Min_F1','Kfold_Std_Deviation_F1','Kfold_F1_Elapsed_Time'],index=indices)
                           
dfcvKF10.loc['XGB']=pd.Series({'Kfold_Max_Accuracy': "%.2f%%" % (CV_XGB_KFOLD10_ACCURACY.max()*100),
                         'Kfold_Mean_Accuracy': "%.2f%%" % (CV_XGB_KFOLD10_ACCURACY.mean()*100),
                         'Kfold_Min_Accuracy': "%.2f%%" % (CV_XGB_KFOLD10_ACCURACY.min()*100),
                         'Kfold_Std_Deviation_Accuracy': "%.2f%%" % (CV_XGB_KFOLD10_ACCURACY.std()*100),
                         'Kfold_Accuracy_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD10_ACCURACY_ELAPSED_TIME,
                         'Kfold_Max_Precision': "%.2f%%" % (CV_XGB_KFOLD10_PRECISION.max()*100),
                         'Kfold_Mean_Precision': "%.2f%%" % (CV_XGB_KFOLD10_PRECISION.mean()*100),
                         'Kfold_Min_Precision': "%.2f%%" % (CV_XGB_KFOLD10_PRECISION.min()*100),
                         'Kfold_Std_Deviation_Precision': "%.2f%%" % (CV_XGB_KFOLD10_PRECISION.std()*100),
                         'Kfold_Precision_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD10_PRECISION_ELAPSED_TIME,
                         'Kfold_Max_Recall': "%.2f%%" % (CV_XGB_KFOLD10_RECALL.max()*100),
                         'Kfold_Mean_Recall': "%.2f%%" % (CV_XGB_KFOLD10_RECALL.mean()*100),
                         'Kfold_Min_Recall': "%.2f%%" % (CV_XGB_KFOLD10_RECALL.min()*100),
                         'Kfold_Std_Deviation_Recall': "%.2f%%" % (CV_XGB_KFOLD10_RECALL.std()*100),
                         'Kfold_Recall_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD10_RECALL_ELAPSED_TIME,
                         'Kfold_Max_F1': "%.2f%%" % (CV_XGB_KFOLD10_F1.max()*100),
                         'Kfold_Mean_F1': "%.2f%%" % (CV_XGB_KFOLD10_F1.mean()*100),
                         'Kfold_Min_F1': "%.2f%%" % (CV_XGB_KFOLD10_F1.min()*100),
                         'Kfold_Std_Deviation_F1': "%.2f%%" % (CV_XGB_KFOLD10_F1.std()*100),
                         'Kfold_F1_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD10_F1_ELAPSED_TIME
                         })

dfcvKF10.loc['DTC']=pd.Series({'Kfold_Max_Accuracy': "%.2f%%" % (CV_DTC_KFOLD10_ACCURACY.max()*100),
                         'Kfold_Mean_Accuracy': "%.2f%%" % (CV_DTC_KFOLD10_ACCURACY.mean()*100),
                         'Kfold_Min_Accuracy': "%.2f%%" % (CV_DTC_KFOLD10_ACCURACY.min()*100),
                         'Kfold_Std_Deviation_Accuracy': "%.2f%%" % (CV_DTC_KFOLD10_ACCURACY.std()*100),
                         'Kfold_Accuracy_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD10_ACCURACY_ELAPSED_TIME,
                         'Kfold_Max_Precision': "%.2f%%" % (CV_DTC_KFOLD10_PRECISION.max()*100),
                         'Kfold_Mean_Precision': "%.2f%%" % (CV_DTC_KFOLD10_PRECISION.mean()*100),
                         'Kfold_Min_Precision': "%.2f%%" % (CV_DTC_KFOLD10_PRECISION.min()*100),
                         'Kfold_Std_Deviation_Precision': "%.2f%%" % (CV_DTC_KFOLD10_PRECISION.std()*100),
                         'Kfold_Precision_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD10_PRECISION_ELAPSED_TIME,
                         'Kfold_Max_Recall': "%.2f%%" % (CV_DTC_KFOLD10_RECALL.max()*100),
                         'Kfold_Mean_Recall': "%.2f%%" % (CV_DTC_KFOLD10_RECALL.mean()*100),
                         'Kfold_Min_Recall': "%.2f%%" % (CV_DTC_KFOLD10_RECALL.min()*100),
                         'Kfold_Std_Deviation_Recall': "%.2f%%" % (CV_DTC_KFOLD10_RECALL.std()*100),
                         'Kfold_Recall_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD10_RECALL_ELAPSED_TIME,
                         'Kfold_Max_F1': "%.2f%%" % (CV_DTC_KFOLD10_F1.max()*100),
                         'Kfold_Mean_F1': "%.2f%%" % (CV_DTC_KFOLD10_F1.mean()*100),
                         'Kfold_Min_F1': "%.2f%%" % (CV_DTC_KFOLD10_F1.min()*100),
                         'Kfold_Std_Deviation_F1': "%.2f%%" % (CV_DTC_KFOLD10_F1.std()*100),
                         'Kfold_F1_Elapsed_Time':"%.2f ms" % CV_DTC_KFOLD10_F1_ELAPSED_TIME
                         })

dfcvKF10.loc['SVC']=pd.Series({'Kfold_Max_Accuracy': "%.2f%%" % (CV_SVC_KFOLD10_ACCURACY.max()*100),
                         'Kfold_Mean_Accuracy': "%.2f%%" % (CV_SVC_KFOLD10_ACCURACY.mean()*100),
                         'Kfold_Min_Accuracy': "%.2f%%" % (CV_SVC_KFOLD10_ACCURACY.min()*100),
                         'Kfold_Std_Deviation_Accuracy': "%.2f%%" % (CV_SVC_KFOLD10_ACCURACY.std()*100),
                         'Kfold_Accuracy_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD10_ACCURACY_ELAPSED_TIME,
                         'Kfold_Max_Precision': "%.2f%%" % (CV_SVC_KFOLD10_PRECISION.max()*100),
                         'Kfold_Mean_Precision': "%.2f%%" % (CV_SVC_KFOLD10_PRECISION.mean()*100),
                         'Kfold_Min_Precision': "%.2f%%" % (CV_SVC_KFOLD10_PRECISION.min()*100),
                         'Kfold_Std_Deviation_Precision': "%.2f%%" % (CV_SVC_KFOLD10_PRECISION.std()*100),
                         'Kfold_Precision_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD10_PRECISION_ELAPSED_TIME,
                         'Kfold_Max_Recall': "%.2f%%" % (CV_SVC_KFOLD10_RECALL.max()*100),
                         'Kfold_Mean_Recall': "%.2f%%" % (CV_SVC_KFOLD10_RECALL.mean()*100),
                         'Kfold_Min_Recall': "%.2f%%" % (CV_SVC_KFOLD10_RECALL.min()*100),
                         'Kfold_Std_Deviation_Recall': "%.2f%%" % (CV_SVC_KFOLD10_RECALL.std()*100),
                         'Kfold_Recall_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD10_RECALL_ELAPSED_TIME,
                         'Kfold_Max_F1': "%.2f%%" % (CV_SVC_KFOLD10_F1.max()*100),
                         'Kfold_Mean_F1': "%.2f%%" % (CV_SVC_KFOLD10_F1.mean()*100),
                         'Kfold_Min_F1': "%.2f%%" % (CV_SVC_KFOLD10_F1.min()*100),
                         'Kfold_Std_Deviation_F1': "%.2f%%" % (CV_SVC_KFOLD10_F1.std()*100),
                         'Kfold_F1_Elapsed_Time':"%.2f ms" % CV_SVC_KFOLD10_F1_ELAPSED_TIME
                         })


df.to_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Classification_Evaluation_Results_NE.csv")
dfcvKF3.to_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Classification_Evaluation_Results_NE_CVKFOLD3.csv")
dfcvKF5.to_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Classification_Evaluation_Results_NE_CVKFOLD5.csv")
dfcvKF10.to_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Classification_Evaluation_Results_NE_CVKFOLD10.csv")
read_file = pd.read_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Classification_Evaluation_Results_NE.csv")
read_file2 = pd.read_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Classification_Evaluation_Results_NE_CVKFOLD3.csv")
read_file3 = pd.read_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Classification_Evaluation_Results_NE_CVKFOLD5.csv")
read_file4 = pd.read_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Classification_Evaluation_Results_NE_CVKFOLD10.csv")

with pd.ExcelWriter('03 - Results/03.1 - NoEncoding/XLSX/Students_Math_Classification_Evaluation_Results_NE.xlsx') as writer:
    read_file.to_excel(writer, index=None, header=True, sheet_name="NoCrossValidation",)
    read_file2.to_excel(writer, index=None, header=True, sheet_name="CrossValidationKFold3")
    read_file3.to_excel(writer, index=None, header=True, sheet_name="CrossValidationKFold5")
    read_file4.to_excel(writer, index=None, header=True, sheet_name="CrossValidationKFold10")