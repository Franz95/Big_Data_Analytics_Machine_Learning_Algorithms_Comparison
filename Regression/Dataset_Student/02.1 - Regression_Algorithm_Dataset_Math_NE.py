from sklearn.model_selection import KFold, cross_val_score, train_test_split 
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from timeit import default_timer as timer
import xgboost as xgb
import pandas as pd
import numpy as np

data= pd.read_csv("02 - Elaborated Dataset/02.1 - NoEncoding/Dataset_Student_Math_NE.csv")

#Dichiariamo i modelli che utilizzeremo

modelXGB=xgb.XGBRegressor()
modelSVR=SVR()
modelDTR=DecisionTreeRegressor()

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
modelDTR.fit(X_train, y_train)
end=timer()
DTR_FITTING_ELAPSED_TIME = round(end - start,2)*1000

start=timer()
modelSVR.fit(X_train, y_train)
end=timer()
SVR_FITTING_ELAPSED_TIME=round(end - start,2)*1000

start=timer()
y_predsXGB = modelXGB.predict(X_test)
predictionsXGB=[round(value) for value in y_predsXGB]
r2XGB= r2_score(y_test,predictionsXGB)
mseXGB=mean_squared_error(y_test,predictionsXGB)
rmseXGB=np.sqrt(mseXGB)
maeXGB=mean_absolute_error(y_test,predictionsXGB)
end=timer()
XGB_EVALUATIONS_ELAPSED_TIME = round(end - start,2)*1000

start=timer()
y_predsDTR = modelDTR.predict(X_test)
predictionsDTR=[round(value) for value in y_predsDTR]
r2DTR= r2_score(y_test,predictionsDTR)
mseDTR=mean_squared_error(y_test,predictionsDTR)
rmseDTR=np.sqrt(mseDTR)
maeDTR=mean_absolute_error(y_test,predictionsDTR)
end=timer()
DTR_EVALUATIONS_ELAPSED_TIME = round(end - start,2)*1000

start=timer()
y_predsSVR = modelSVR.predict(X_test)
predictionsSVR=[round(value) for value in y_predsSVR]
r2SVR= r2_score(y_test,predictionsSVR)
mseSVR=mean_squared_error(y_test,predictionsSVR)
rmseSVR=np.sqrt(mseSVR)
maeSVR=mean_absolute_error(y_test,predictionsSVR)
end=timer()
SVR_EVALUATIONS_ELAPSED_TIME = round(end - start,2)*1000

#CROSS VALIDATION 3-SPLIT
kfold=KFold(n_splits=3, shuffle=True, random_state=9)
scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']

start=timer()
CV_XGB_KFOLD3_R2= cross_val_score(modelXGB,X,y, cv=kfold, scoring='r2')
end=timer()
CV_XGB_KFOLD3_R2_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD3_R2= cross_val_score(modelDTR,X,y, cv=kfold, scoring='r2')
end=timer()
CV_DTR_KFOLD3_R2_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD3_R2= cross_val_score(modelSVR,X,y, cv=kfold, scoring='r2')
end=timer()
CV_SVR_KFOLD3_R2_ELAPSED_TIME= round(end - start, 2)*1000


start=timer()
CV_XGB_KFOLD3_MEAN_SQUARED_ERROR= -cross_val_score(modelXGB,X,y, cv=kfold, scoring='neg_mean_squared_error')
end=timer()
CV_XGB_KFOLD3_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD3_MEAN_SQUARED_ERROR= -cross_val_score(modelDTR,X,y, cv=kfold, scoring='neg_mean_squared_error')
end=timer()
CV_DTR_KFOLD3_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD3_MEAN_SQUARED_ERROR= -cross_val_score(modelSVR,X,y, cv=kfold, scoring='neg_mean_squared_error')
end=timer()
CV_SVR_KFOLD3_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000


start=timer()
CV_XGB_KFOLD3_MEAN_ABSOLUTE_ERROR= -cross_val_score(modelXGB,X,y, cv=kfold, scoring='neg_mean_absolute_error')
end=timer()
CV_XGB_KFOLD3_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD3_MEAN_ABSOLUTE_ERROR= -cross_val_score(modelDTR,X,y, cv=kfold, scoring='neg_mean_absolute_error')
end=timer()
CV_DTR_KFOLD3_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD3_MEAN_ABSOLUTE_ERROR= -cross_val_score(modelSVR,X,y, cv=kfold, scoring='neg_mean_absolute_error')
end=timer()
CV_SVR_KFOLD3_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME= round(end - start, 2)*1000


start=timer()
CV_XGB_KFOLD3_ROOT_MEAN_SQUARED_ERROR= -cross_val_score(modelXGB,X,y, cv=kfold, scoring='neg_root_mean_squared_error')
end=timer()
CV_XGB_KFOLD3_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD3_ROOT_MEAN_SQUARED_ERROR= -cross_val_score(modelDTR,X,y, cv=kfold, scoring='neg_root_mean_squared_error')
end=timer()
CV_DTR_KFOLD3_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD3_ROOT_MEAN_SQUARED_ERROR= -cross_val_score(modelSVR,X,y, cv=kfold, scoring='neg_root_mean_squared_error')
end=timer()
CV_SVR_KFOLD3_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

#CROSS VALIDATION 5-SPLIT
kfold=KFold(n_splits=5, shuffle=True, random_state=9)
scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']

start=timer()
CV_XGB_KFOLD5_R2= cross_val_score(modelXGB,X,y, cv=kfold, scoring='r2')
end=timer()
CV_XGB_KFOLD5_R2_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD5_R2= cross_val_score(modelDTR,X,y, cv=kfold, scoring='r2')
end=timer()
CV_DTR_KFOLD5_R2_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD5_R2= cross_val_score(modelSVR,X,y, cv=kfold, scoring='r2')
end=timer()
CV_SVR_KFOLD5_R2_ELAPSED_TIME= round(end - start, 2)*1000


start=timer()
CV_XGB_KFOLD5_MEAN_SQUARED_ERROR= -cross_val_score(modelXGB,X,y, cv=kfold, scoring='neg_mean_squared_error')
end=timer()
CV_XGB_KFOLD5_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD5_MEAN_SQUARED_ERROR= -cross_val_score(modelDTR,X,y, cv=kfold, scoring='neg_mean_squared_error')
end=timer()
CV_DTR_KFOLD5_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD5_MEAN_SQUARED_ERROR= -cross_val_score(modelSVR,X,y, cv=kfold, scoring='neg_mean_squared_error')
end=timer()
CV_SVR_KFOLD5_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000


start=timer()
CV_XGB_KFOLD5_MEAN_ABSOLUTE_ERROR= -cross_val_score(modelXGB,X,y, cv=kfold, scoring='neg_mean_absolute_error')
end=timer()
CV_XGB_KFOLD5_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD5_MEAN_ABSOLUTE_ERROR= -cross_val_score(modelDTR,X,y, cv=kfold, scoring='neg_mean_absolute_error')
end=timer()
CV_DTR_KFOLD5_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD5_MEAN_ABSOLUTE_ERROR= -cross_val_score(modelSVR,X,y, cv=kfold, scoring='neg_mean_absolute_error')
end=timer()
CV_SVR_KFOLD5_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME= round(end - start, 2)*1000


start=timer()
CV_XGB_KFOLD5_ROOT_MEAN_SQUARED_ERROR= -cross_val_score(modelXGB,X,y, cv=kfold, scoring='neg_root_mean_squared_error')
end=timer()
CV_XGB_KFOLD5_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD5_ROOT_MEAN_SQUARED_ERROR= -cross_val_score(modelDTR,X,y, cv=kfold, scoring='neg_root_mean_squared_error')
end=timer()
CV_DTR_KFOLD5_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD5_ROOT_MEAN_SQUARED_ERROR= -cross_val_score(modelSVR,X,y, cv=kfold, scoring='neg_root_mean_squared_error')
end=timer()
CV_SVR_KFOLD5_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

#CROSS VALIDATION 10-SPLIT
kfold=KFold(n_splits=10, shuffle=True, random_state=9)
scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']

start=timer()
CV_XGB_KFOLD10_R2= cross_val_score(modelXGB,X,y, cv=kfold, scoring='r2')
end=timer()
CV_XGB_KFOLD10_R2_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD10_R2= cross_val_score(modelDTR,X,y, cv=kfold, scoring='r2')
end=timer()
CV_DTR_KFOLD10_R2_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD10_R2= cross_val_score(modelSVR,X,y, cv=kfold, scoring='r2')
end=timer()
CV_SVR_KFOLD10_R2_ELAPSED_TIME= round(end - start, 2)*1000


start=timer()
CV_XGB_KFOLD10_MEAN_SQUARED_ERROR= -cross_val_score(modelXGB,X,y, cv=kfold, scoring='neg_mean_squared_error')
end=timer()
CV_XGB_KFOLD10_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD10_MEAN_SQUARED_ERROR= -cross_val_score(modelDTR,X,y, cv=kfold, scoring='neg_mean_squared_error')
end=timer()
CV_DTR_KFOLD10_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD10_MEAN_SQUARED_ERROR= -cross_val_score(modelSVR,X,y, cv=kfold, scoring='neg_mean_squared_error')
end=timer()
CV_SVR_KFOLD10_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000


start=timer()
CV_XGB_KFOLD10_MEAN_ABSOLUTE_ERROR= -cross_val_score(modelXGB,X,y, cv=kfold, scoring='neg_mean_absolute_error')
end=timer()
CV_XGB_KFOLD10_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD10_MEAN_ABSOLUTE_ERROR= -cross_val_score(modelDTR,X,y, cv=kfold, scoring='neg_mean_absolute_error')
end=timer()
CV_DTR_KFOLD10_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD10_MEAN_ABSOLUTE_ERROR= -cross_val_score(modelSVR,X,y, cv=kfold, scoring='neg_mean_absolute_error')
end=timer()
CV_SVR_KFOLD10_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME= round(end - start, 2)*1000


start=timer()
CV_XGB_KFOLD10_ROOT_MEAN_SQUARED_ERROR= -cross_val_score(modelXGB,X,y, cv=kfold, scoring='neg_root_mean_squared_error')
end=timer()
CV_XGB_KFOLD10_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_DTR_KFOLD10_ROOT_MEAN_SQUARED_ERROR= -cross_val_score(modelDTR,X,y, cv=kfold, scoring='neg_root_mean_squared_error')
end=timer()
CV_DTR_KFOLD10_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

start=timer()
CV_SVR_KFOLD10_ROOT_MEAN_SQUARED_ERROR= -cross_val_score(modelSVR,X,y, cv=kfold, scoring='neg_root_mean_squared_error')
end=timer()
CV_SVR_KFOLD10_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME= round(end - start, 2)*1000

#GENERIAMO I CSV E GLI EXCEL CON TUTTI I RISULTATI RACCOLTI DALLE ANALISI EFFETTUATE
indices=['XGB','DTR','SVR']
df = pd.DataFrame(columns=['r2','Mean_Squared_Error','Mean_Absolute_Error','Root_Mean_Squared_Error','Fit_Elapsed_Time','Evaluation_Elapsed_Time'], index=indices)
                           
df.loc['XGB']=pd.Series({'r2': "%.2f" % (r2XGB),
                         'Mean_Squared_Error': "%.2f" % (mseXGB),
                         'Mean_Absolute_Error': "%.2f" % (maeXGB),
                         'Root_Mean_Squared_Error': "%.2f" % (rmseXGB),
                         'Fit_Elapsed_Time':"%.2f ms" % XGB_FITTING_ELAPSED_TIME,
                         'Evaluation_Elapsed_Time':"%.2f ms" % XGB_EVALUATIONS_ELAPSED_TIME
                         })

df.loc['DTR']=pd.Series({'r2': "%.2f" % (r2DTR),
                         'Mean_Squared_Error': "%.2f" % (mseDTR),
                         'Mean_Absolute_Error': "%.2f" % (maeDTR),
                         'Root_Mean_Squared_Error': "%.2f" % (rmseDTR),
                         'Fit_Elapsed_Time':"%.2f ms" % DTR_FITTING_ELAPSED_TIME,
                         'Evaluation_Elapsed_Time':"%.2f ms" % DTR_EVALUATIONS_ELAPSED_TIME
                         })

df.loc['SVR']=pd.Series({'r2': "%.2f" % (r2SVR),
                         'Mean_Squared_Error': "%.2f" % (mseSVR),
                         'Mean_Absolute_Error': "%.2f" % (maeSVR),
                         'Root_Mean_Squared_Error': "%.2f" % (rmseSVR),
                         'Fit_Elapsed_Time':"%.2f ms" % SVR_FITTING_ELAPSED_TIME,
                         'Evaluation_Elapsed_Time':"%.2f ms" % SVR_EVALUATIONS_ELAPSED_TIME
                         })

dfcvKF3 = pd.DataFrame(columns=['Kfold_Max_r2','Kfold_Mean_r2','Kfold_Min_r2','Kfold_Std_Deviation_r2','Kfold_r2_Elapsed_Time',
                           'Kfold_Max_Mean_Squared_Error','Kfold_Mean_Mean_Squared_Error','Kfold_Min_Mean_Squared_Error','Kfold_Std_Deviation_Mean_Squared_Error','Kfold_Mean_Squared_Error_Elapsed_Time',
                           'Kfold_Max_Mean_Absolute_Error','Kfold_Mean_Mean_Absolute_Error','Kfold_Min_Mean_Absolute_Error','Kfold_Std_Deviation_Mean_Absolute_Error','Kfold_Mean_Absolute_Error_Elapsed_Time',
                           'Kfold_Max_Root_Mean_Squared_Error','Kfold_Mean_Root_Mean_Squared_Error','Kfold_Min_Root_Mean_Squared_Error','Kfold_Std_Deviation_Root_Mean_Squared_Error','Kfold_Root_Mean_Squared_Error_Elapsed_Time'],index=indices)
                           
dfcvKF3.loc['XGB']=pd.Series({'Kfold_Max_r2': "%.2f" % (CV_XGB_KFOLD3_R2.max()),
                         'Kfold_Mean_r2': "%.2f" % (CV_XGB_KFOLD3_R2.mean()),
                         'Kfold_Min_r2': "%.2f" % (CV_XGB_KFOLD3_R2.min()),
                         'Kfold_Std_Deviation_r2': "%.2f" % (CV_XGB_KFOLD3_R2.std()),
                         'Kfold_r2_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD3_R2_ELAPSED_TIME,
                         'Kfold_Max_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD3_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD3_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD3_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD3_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD3_MEAN_SQUARED_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD3_MEAN_ABSOLUTE_ERROR.max()),
                         'Kfold_Mean_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD3_MEAN_ABSOLUTE_ERROR.mean()),
                         'Kfold_Min_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD3_MEAN_ABSOLUTE_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD3_MEAN_ABSOLUTE_ERROR.std()),
                         'Kfold_Mean_Absolute_Error_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD3_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD3_ROOT_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD3_ROOT_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD3_ROOT_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD3_ROOT_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Root_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD3_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME
                         })

dfcvKF3.loc['DTR']=pd.Series({'Kfold_Max_r2': "%.2f" % (CV_DTR_KFOLD3_R2.max()),
                         'Kfold_Mean_r2': "%.2f" % (CV_DTR_KFOLD3_R2.mean()),
                         'Kfold_Min_r2': "%.2f" % (CV_DTR_KFOLD3_R2.min()),
                         'Kfold_Std_Deviation_r2': "%.2f" % (CV_DTR_KFOLD3_R2.std()),
                         'Kfold_r2_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD3_R2_ELAPSED_TIME,
                         'Kfold_Max_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD3_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD3_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD3_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD3_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD3_MEAN_SQUARED_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD3_MEAN_ABSOLUTE_ERROR.max()),
                         'Kfold_Mean_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD3_MEAN_ABSOLUTE_ERROR.mean()),
                         'Kfold_Min_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD3_MEAN_ABSOLUTE_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD3_MEAN_ABSOLUTE_ERROR.std()),
                         'Kfold_Mean_Absolute_Error_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD3_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD3_ROOT_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD3_ROOT_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD3_ROOT_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD3_ROOT_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Root_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD3_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME
                         })

dfcvKF3.loc['SVR']=pd.Series({'Kfold_Max_r2': "%.2f" % (CV_SVR_KFOLD3_R2.max()),
                         'Kfold_Mean_r2': "%.2f" % (CV_SVR_KFOLD3_R2.mean()),
                         'Kfold_Min_r2': "%.2f" % (CV_SVR_KFOLD3_R2.min()),
                         'Kfold_Std_Deviation_r2': "%.2f" % (CV_SVR_KFOLD3_R2.std()),
                         'Kfold_r2_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD3_R2_ELAPSED_TIME,
                         'Kfold_Max_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD3_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD3_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD3_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD3_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD3_MEAN_SQUARED_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD3_MEAN_ABSOLUTE_ERROR.max()),
                         'Kfold_Mean_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD3_MEAN_ABSOLUTE_ERROR.mean()),
                         'Kfold_Min_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD3_MEAN_ABSOLUTE_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD3_MEAN_ABSOLUTE_ERROR.std()),
                         'Kfold_Mean_Absolute_Error_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD3_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD3_ROOT_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD3_ROOT_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD3_ROOT_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD3_ROOT_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Root_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD3_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME
                         })

dfcvKF5 = pd.DataFrame(columns=['Kfold_Max_r2','Kfold_Mean_r2','Kfold_Min_r2','Kfold_Std_Deviation_r2','Kfold_r2_Elapsed_Time',
                           'Kfold_Max_Mean_Squared_Error','Kfold_Mean_Mean_Squared_Error','Kfold_Min_Mean_Squared_Error','Kfold_Std_Deviation_Mean_Squared_Error','Kfold_Mean_Squared_Error_Elapsed_Time',
                           'Kfold_Max_Mean_Absolute_Error','Kfold_Mean_Mean_Absolute_Error','Kfold_Min_Mean_Absolute_Error','Kfold_Std_Deviation_Mean_Absolute_Error','Kfold_Mean_Absolute_Error_Elapsed_Time',
                           'Kfold_Max_Root_Mean_Squared_Error','Kfold_Mean_Root_Mean_Squared_Error','Kfold_Min_Root_Mean_Squared_Error','Kfold_Std_Deviation_Root_Mean_Squared_Error','Kfold_Root_Mean_Squared_Error_Elapsed_Time'],index=indices)
                           
dfcvKF5.loc['XGB']=pd.Series({'Kfold_Max_r2': "%.2f" % (CV_XGB_KFOLD5_R2.max()),
                         'Kfold_Mean_r2': "%.2f" % (CV_XGB_KFOLD5_R2.mean()),
                         'Kfold_Min_r2': "%.2f" % (CV_XGB_KFOLD5_R2.min()),
                         'Kfold_Std_Deviation_r2': "%.2f" % (CV_XGB_KFOLD5_R2.std()),
                         'Kfold_r2_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD5_R2_ELAPSED_TIME,
                         'Kfold_Max_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD5_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD5_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD5_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD5_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD5_MEAN_SQUARED_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD5_MEAN_ABSOLUTE_ERROR.max()),
                         'Kfold_Mean_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD5_MEAN_ABSOLUTE_ERROR.mean()),
                         'Kfold_Min_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD5_MEAN_ABSOLUTE_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD5_MEAN_ABSOLUTE_ERROR.std()),
                         'Kfold_Mean_Absolute_Error_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD5_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD5_ROOT_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD5_ROOT_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD5_ROOT_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD5_ROOT_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Root_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD5_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME
                         })

dfcvKF5.loc['DTR']=pd.Series({'Kfold_Max_r2': "%.2f" % (CV_DTR_KFOLD5_R2.max()),
                         'Kfold_Mean_r2': "%.2f" % (CV_DTR_KFOLD5_R2.mean()),
                         'Kfold_Min_r2': "%.2f" % (CV_DTR_KFOLD5_R2.min()),
                         'Kfold_Std_Deviation_r2': "%.2f" % (CV_DTR_KFOLD5_R2.std()),
                         'Kfold_r2_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD5_R2_ELAPSED_TIME,
                         'Kfold_Max_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD5_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD5_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD5_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD5_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD5_MEAN_SQUARED_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD5_MEAN_ABSOLUTE_ERROR.max()),
                         'Kfold_Mean_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD5_MEAN_ABSOLUTE_ERROR.mean()),
                         'Kfold_Min_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD5_MEAN_ABSOLUTE_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD5_MEAN_ABSOLUTE_ERROR.std()),
                         'Kfold_Mean_Absolute_Error_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD5_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD5_ROOT_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD5_ROOT_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD5_ROOT_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD5_ROOT_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Root_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD5_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME
                         })

dfcvKF5.loc['SVR']=pd.Series({'Kfold_Max_r2': "%.2f" % (CV_SVR_KFOLD5_R2.max()),
                         'Kfold_Mean_r2': "%.2f" % (CV_SVR_KFOLD5_R2.mean()),
                         'Kfold_Min_r2': "%.2f" % (CV_SVR_KFOLD5_R2.min()),
                         'Kfold_Std_Deviation_r2': "%.2f" % (CV_SVR_KFOLD5_R2.std()),
                         'Kfold_r2_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD5_R2_ELAPSED_TIME,
                         'Kfold_Max_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD5_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD5_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD5_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD5_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD5_MEAN_SQUARED_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD5_MEAN_ABSOLUTE_ERROR.max()),
                         'Kfold_Mean_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD5_MEAN_ABSOLUTE_ERROR.mean()),
                         'Kfold_Min_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD5_MEAN_ABSOLUTE_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD5_MEAN_ABSOLUTE_ERROR.std()),
                         'Kfold_Mean_Absolute_Error_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD5_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD5_ROOT_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD5_ROOT_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD5_ROOT_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD5_ROOT_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Root_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD5_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME
                         })

dfcvKF10 = pd.DataFrame(columns=['Kfold_Max_r2','Kfold_Mean_r2','Kfold_Min_r2','Kfold_Std_Deviation_r2','Kfold_r2_Elapsed_Time',
                           'Kfold_Max_Mean_Squared_Error','Kfold_Mean_Mean_Squared_Error','Kfold_Min_Mean_Squared_Error','Kfold_Std_Deviation_Mean_Squared_Error','Kfold_Mean_Squared_Error_Elapsed_Time',
                           'Kfold_Max_Mean_Absolute_Error','Kfold_Mean_Mean_Absolute_Error','Kfold_Min_Mean_Absolute_Error','Kfold_Std_Deviation_Mean_Absolute_Error','Kfold_Mean_Absolute_Error_Elapsed_Time',
                           'Kfold_Max_Root_Mean_Squared_Error','Kfold_Mean_Root_Mean_Squared_Error','Kfold_Min_Root_Mean_Squared_Error','Kfold_Std_Deviation_Root_Mean_Squared_Error','Kfold_Root_Mean_Squared_Error_Elapsed_Time'],index=indices)
                           
dfcvKF10.loc['XGB']=pd.Series({'Kfold_Max_r2': "%.2f" % (CV_XGB_KFOLD10_R2.max()),
                         'Kfold_Mean_r2': "%.2f" % (CV_XGB_KFOLD10_R2.mean()),
                         'Kfold_Min_r2': "%.2f" % (CV_XGB_KFOLD10_R2.min()),
                         'Kfold_Std_Deviation_r2': "%.2f" % (CV_XGB_KFOLD10_R2.std()),
                         'Kfold_r2_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD10_R2_ELAPSED_TIME,
                         'Kfold_Max_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD10_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD10_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD10_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD10_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD10_MEAN_SQUARED_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD10_MEAN_ABSOLUTE_ERROR.max()),
                         'Kfold_Mean_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD10_MEAN_ABSOLUTE_ERROR.mean()),
                         'Kfold_Min_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD10_MEAN_ABSOLUTE_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Absolute_Error': "%.2f" % (CV_XGB_KFOLD10_MEAN_ABSOLUTE_ERROR.std()),
                         'Kfold_Mean_Absolute_Error_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD10_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD10_ROOT_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD10_ROOT_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD10_ROOT_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Root_Mean_Squared_Error': "%.2f" % (CV_XGB_KFOLD10_ROOT_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Root_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_XGB_KFOLD10_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME
                         })

dfcvKF10.loc['DTR']=pd.Series({'Kfold_Max_r2': "%.2f" % (CV_DTR_KFOLD10_R2.max()),
                         'Kfold_Mean_r2': "%.2f" % (CV_DTR_KFOLD10_R2.mean()),
                         'Kfold_Min_r2': "%.2f" % (CV_DTR_KFOLD10_R2.min()),
                         'Kfold_Std_Deviation_r2': "%.2f" % (CV_DTR_KFOLD10_R2.std()),
                         'Kfold_r2_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD10_R2_ELAPSED_TIME,
                         'Kfold_Max_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD10_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD10_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD10_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD10_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD10_MEAN_SQUARED_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD10_MEAN_ABSOLUTE_ERROR.max()),
                         'Kfold_Mean_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD10_MEAN_ABSOLUTE_ERROR.mean()),
                         'Kfold_Min_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD10_MEAN_ABSOLUTE_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Absolute_Error': "%.2f" % (CV_DTR_KFOLD10_MEAN_ABSOLUTE_ERROR.std()),
                         'Kfold_Mean_Absolute_Error_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD10_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD10_ROOT_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD10_ROOT_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD10_ROOT_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Root_Mean_Squared_Error': "%.2f" % (CV_DTR_KFOLD10_ROOT_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Root_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_DTR_KFOLD10_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME
                         })

dfcvKF10.loc['SVR']=pd.Series({'Kfold_Max_r2': "%.2f" % (CV_SVR_KFOLD10_R2.max()),
                         'Kfold_Mean_r2': "%.2f" % (CV_SVR_KFOLD10_R2.mean()),
                         'Kfold_Min_r2': "%.2f" % (CV_SVR_KFOLD10_R2.min()),
                         'Kfold_Std_Deviation_r2': "%.2f" % (CV_SVR_KFOLD10_R2.std()),
                         'Kfold_r2_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD10_R2_ELAPSED_TIME,
                         'Kfold_Max_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD10_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD10_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD10_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD10_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD10_MEAN_SQUARED_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD10_MEAN_ABSOLUTE_ERROR.max()),
                         'Kfold_Mean_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD10_MEAN_ABSOLUTE_ERROR.mean()),
                         'Kfold_Min_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD10_MEAN_ABSOLUTE_ERROR.min()),
                         'Kfold_Std_Deviation_Mean_Absolute_Error': "%.2f" % (CV_SVR_KFOLD10_MEAN_ABSOLUTE_ERROR.std()),
                         'Kfold_Mean_Absolute_Error_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD10_MEAN_ABSOLUTE_ERROR_ELAPSED_TIME,
                         'Kfold_Max_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD10_ROOT_MEAN_SQUARED_ERROR.max()),
                         'Kfold_Mean_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD10_ROOT_MEAN_SQUARED_ERROR.mean()),
                         'Kfold_Min_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD10_ROOT_MEAN_SQUARED_ERROR.min()),
                         'Kfold_Std_Deviation_Root_Mean_Squared_Error': "%.2f" % (CV_SVR_KFOLD10_ROOT_MEAN_SQUARED_ERROR.std()),
                         'Kfold_Root_Mean_Squared_Error_Elapsed_Time':"%.2f ms" % CV_SVR_KFOLD10_ROOT_MEAN_SQUARED_ERROR_ELAPSED_TIME
                         })

df.to_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Regression_Evaluation_Results_NE.csv")
dfcvKF3.to_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Regression_Evaluation_Results_NE_CVKFOLD3.csv")
dfcvKF5.to_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Regression_Evaluation_Results_NE_CVKFOLD5.csv")
dfcvKF10.to_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Regression_Evaluation_Results_NE_CVKFOLD10.csv")

read_file = pd.read_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Regression_Evaluation_Results_NE.csv")
read_file2 = pd.read_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Regression_Evaluation_Results_NE_CVKFOLD3.csv")
read_file3 = pd.read_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Regression_Evaluation_Results_NE_CVKFOLD5.csv")
read_file4 = pd.read_csv("03 - Results/03.1 - NoEncoding/CSV/Students_Math_Regression_Evaluation_Results_NE_CVKFOLD10.csv")

with pd.ExcelWriter('03 - Results/03.1 - NoEncoding/XLSX/Students_Math_Regression_Evaluation_Results_NE.xlsx') as writer:
    read_file.to_excel(writer, index=None, header=True, sheet_name="NoCrossValidation",)
    read_file2.to_excel(writer, index=None, header=True, sheet_name="CrossValidationKFold3")
    read_file3.to_excel(writer, index=None, header=True, sheet_name="CrossValidationKFold5")
    read_file4.to_excel(writer, index=None, header=True, sheet_name="CrossValidationKFold10")