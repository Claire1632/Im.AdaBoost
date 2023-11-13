import numpy as np
import pandas as pd
from data import Vertebral_column
#from data import seismic_bumps
#from data import churn
#from data import indian_liver_patient
#from data import spect_heart
import svm
#from sklearn.metrics import classification_report
import trainning_of_adaboost as toa
from sklearn.ensemble import AdaBoostClassifier
import adaboost_svm
from report import report
#from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score
#test SVM

C = [] #mang C
for i in range(50,20000,10):
    C.append(i)

    #Vertebral_column
m = []
Name = "Vertibral_column, SVM, test_size.txt"
f = open(Name, "w")
for l in range(3,21,2):

    for i in range(3, 6):
        f_ngon = 0
        C_ngon = 0
        C_tong = 0
        C_tb = 0
        for o in range(1,11):
            for j in C:
                print (j)
                print ("\n")
                X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate = 1/l)
                #X_train, y_train, X_test, y_test = seismic_bumps.load_data(test_size= i * 0.1, new_rate = 1/17)
                #learner : SVM
                #f.write("\n\n learnerSVM: test size = %s, C = %s \n\n" %(i*0.1,j))    
                w, b = svm.fit(X_train, y_train, j)
                test_pred = np.sign(X_test.dot(w)+b)
                precision, recall, fscore, support = score(y_test, test_pred)
                #m.append(f1)
                #print (fscore[1])
                if(fscore[1]>=f_ngon):
                    f_ngon = fscore[1]
                    C_ngon = j
                
                #f.write(report(y_test,test_pred).to_string())
            C_tong += C_ngon
        C_tb = C_tong/10
        f.write("New rate: 1/%s, Test size: %s, C: %s , f: %s \n"%(l,i,C_tb,f_ngon))
f.close()