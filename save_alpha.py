import numpy as np
import pandas as pd
from data import Vertebral_column
from data import seismic_bumps
from data import churn
from data import indian_liver_patient
from data import spect_heart
import svm
#from sklearn.metrics import classification_report
import trainning_of_adaboost as toa
from sklearn.ensemble import AdaBoostClassifier
import adaboost_svm
from report import report
#from sklearn.metrics import f1_score
from sklearn.metrics  import classification_report,roc_auc_score,precision_recall_fscore_support as score

M=5
C=10000
# theta=2
for theta in range(1,2,1):
    theta=0.2
    for mr in range(3,4,2):
        new_rate = 1/mr
        #Name2 = "0110_n Vertebral column Alpha_Im1.ADABoost"+str(theta)+"_rate_1_"+str(mr)+".txt"
        Name2 = "0110_n Indian liver patient Alpha_Im1.ADABoost_theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        Name = "1808_n Vertebral column "+str(theta)+"_rate_1_"+str(mr)+".txt"
        # Name = "Churn_theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        #Name = "Semic bump _theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        # Name = "spect_heart_theta "+str(theta)+"_rate_1_"+str(mr)+".txt"
        f2 = open(Name2, "w")
        f = open(Name, "w")
        f.write("\n\n Adaboost with SVM (sklearn):")
        print("\n\n Adaboost with SVM (sklearn):\n")
        f2_st="*****test alpha**** \n"

        for i in range(3, 4): 
            S_precision_A_SVM=0
            S_recall_A_SVM=0
            S_fscore_A_SVM=0
            S_precision_A_SVM_N=0
            S_recall_A_SVM_N=0
            S_fscore_A_SVM_N=0

            S_precision_A_WSVM=0
            S_recall_A_WSVM=0
            S_fscore_A_WSVM=0
            S_precision_A_WSVM_N=0
            S_recall_A_WSVM_N=0
            S_fscore_A_WSVM_N=0

            S_precision_A1_SVM=0
            S_recall_A1_SVM=0
            S_fscore_A1_SVM=0
            S_precision_A1_SVM_N=0
            S_recall_A1_SVM_N=0
            S_fscore_A1_SVM_N=0

            S_precision_A1_WSVM=0
            S_recall_A1_WSVM=0
            S_fscore_A1_WSVM=0
            S_precision_A1_WSVM_N=0
            S_recall_A1_WSVM_N=0
            S_fscore_A1_WSVM_N=0

            S_precision_A2_SVM=0
            S_recall_A2_SVM=0
            S_fscore_A2_SVM=0
            S_precision_A2_SVM_N=0
            S_recall_A2_SVM_N=0
            S_fscore_A2_SVM_N=0

            S_precision_A2_WSVM=0
            S_recall_A2_WSVM=0
            S_fscore_A2_WSVM=0
            S_precision_A2_WSVM_N=0
            S_recall_A2_WSVM_N=0
            S_fscore_A2_WSVM_N=0

            S_precision_A12_SVM=0
            S_recall_A12_SVM=0
            S_fscore_A12_SVM=0
            S_precision_A12_SVM_N=0
            S_recall_A12_SVM_N=0
            S_fscore_A12_SVM_N=0

            S_precision_A12_WSVM=0
            S_recall_A12_WSVM=0
            S_fscore_A12_WSVM=0
            S_precision_A12_WSVM_N=0
            S_recall_A12_WSVM_N=0
            S_fscore_A12_WSVM_N=0

            f.write("\n\n New rate = %s, test size = %s, C = %s, M = %s\n\n" %(new_rate,i*0.1, C, M))
            print("\n\n New rate = %s, test size = %s, C = %s, M = %s\n\n" %(new_rate,i*0.1, C, M))
            print("\n Test for 1 to 20: \n")            
            for kf in range(1,10):
                alpha_test=[] #ktra Alpha
                alpha_test_im=[] #ktra Alpha IM1
                alpha_test_im12=[] #ktra Alpha IM12
                X_train = None
                y_train = None
                X_test = None
                y_test = None
                #X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
                # X_train, y_train, X_test, y_test = spect_heart.load_data(test_size= i * 0.1, new_rate=new_rate)
                # X_train, y_train, X_test, y_test = churn.load_data(test_size= i * 0.1, new_rate=new_rate)
                # X_train, y_train, X_test, y_test = seismic_bumps.load_data(test_size= i * 0.1, new_rate=new_rate)
                X_train, y_train, X_test, y_test = indian_liver_patient.load_data(test_size= i * 0.1, new_rate=new_rate)
        #learner : Adaboost with SVM lib``
                print("\n\n Test :"+str(kf)+"\n")
#-----------------------
                # print("ADA Boost with SVM starting...\n")
                # w, b, a = adaboost_svm.fit(X_train, y_train, M, C, instance_categorization=False, proposed = False)
                # test_pred = adaboost_svm.predict(X_test, w, b, a, M)
                # precision, recall, fscore, support = score(y_test, test_pred)
                # S_precision_A_SVM=S_precision_A_SVM+precision[1]
                # S_recall_A_SVM=S_recall_A_SVM+recall[1]
                #
                # S_precision_A_SVM_N=S_precision_A_SVM_N+precision[0]
                # S_recall_A_SVM_N=S_recall_A_SVM_N+recall[0]

                # print("Done.\n")

        # # #learner : Adaboost with W.SVM paper 2016 

                print("ADA Boost with W.SVM starting...\n")
                # # X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
                w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization=True, proposed_preprocessing = False,proposed_alpha = False, test_something = True, theta=theta)
                test_pred2 = toa.predict(X_test, w, b, a, M)
                precision, recall, fscore, support = score(y_test, test_pred2)
                S_precision_A_WSVM=S_precision_A_WSVM+precision[1]
                S_recall_A_WSVM=S_recall_A_WSVM+recall[1]
                
                ## Am
                S_precision_A_WSVM_N=S_precision_A_WSVM_N+precision[0]
                S_recall_A_WSVM_N=S_recall_A_WSVM_N+recall[0]
                #test alpha
                alpha_test.append(a)
                f2_st="theta="+str(theta)+"_rate_1_"+str(mr)+"TestSize_"+str(i)+"loop"+str(kf)+str(alpha_test)+"\n"
                f2.write(f2_st)
                #

                print("Done.\n")
                w = None
                b = None
                a = None

        ###learner : Adaboost with proposed

                # print("ADA Boost Nova 1 with SVM starting...\n")
                # w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization=False, proposed_preprocessing = True, proposed_alpha = False, test_something = True, theta=theta)
                # test_pred3 = toa.predict(X_test, w, b, a, M)
                # precision, recall, fscore, support = score(y_test, test_pred3)
                # S_precision_A1_SVM=S_precision_A1_SVM+precision[1]
                # S_recall_A1_SVM=S_recall_A1_SVM+recall[1]
                
                # #
                # S_precision_A1_SVM_N=S_precision_A1_SVM_N+precision[0]
                # S_recall_A1_SVM_N=S_recall_A1_SVM_N+recall[0]
                # print("Done.\n")
                # w = None
                # b = None
                # a = None

        ###learner : Adaboost Nova 1 with W.SVM  2016

                print("ADA Boost Nova 1 with W.SVM starting...\n")
                # X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
                w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization=True, proposed_preprocessing = True, proposed_alpha = False, test_something = True, theta=theta)
                
                test_pred4 = toa.predict(X_test, w, b, a, M)
                precision, recall, fscore, support = score(y_test, test_pred4)
                S_precision_A1_WSVM=S_precision_A1_WSVM+precision[1]
                S_recall_A1_WSVM=S_recall_A1_WSVM+recall[1]
                #
                S_precision_A1_WSVM_N=S_precision_A1_WSVM_N+precision[0]
                S_recall_A1_WSVM_N=S_recall_A1_WSVM_N+recall[0]
                print("Done.\n")
                

                #test Alpha
                
                alpha_test_im.append(a)
                f2_st_im="theta="+str(theta)+"_rate_1_"+str(mr)+"TestSize_"+str(i)+"loop"+str(kf)+str(alpha_test_im)+"\n\n\n"
                f2.write(f2_st_im)

                #
                w = None
                b = None
                a = None
                
            #learner : Adaboost with thuat toan cai tien alpha

                # print("ADA Boost Nova 2 with SVM starting...\n")
                # # X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
                # w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization=False,proposed_preprocessing = False, proposed_alpha = True, test_something = True, theta=theta)
                # test_pred5 = toa.predict(X_test, w, b, a, M)
                # precision, recall, fscore, support = score(y_test, test_pred5)
                # S_precision_A2_SVM=S_precision_A2_SVM+precision[1]
                # S_recall_A2_SVM=S_recall_A2_SVM+recall[1]
                # #
                # S_precision_A2_SVM_N=S_precision_A2_SVM_N+precision[0]
                # S_recall_A2_SVM_N=S_recall_A2_SVM_N+recall[0]
                # print("Done.\n")
                # w = None
                # b = None
                # a = None

        ### #learner : ADA Boost Nova 2 with W.SVM 2016
                # print("ADA Boost Nova 2 with W.SVM starting...\n")
                # # X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
                # w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization = True,proposed_preprocessing = False, proposed_alpha = True, test_something = True, theta=theta)
                # test_pred6 = toa.predict(X_test, w, b, a, M)
                # precision, recall, fscore, support = score(y_test, test_pred6)
                # S_precision_A2_WSVM=S_precision_A2_WSVM+precision[1]
                # S_recall_A2_WSVM=S_recall_A2_WSVM+recall[1]

                # #
                # S_precision_A2_WSVM_N=S_precision_A2_WSVM_N+precision[0]
                # S_recall_A2_WSVM_N=S_recall_A2_WSVM_N+recall[0]
                # print("Done.\n")
                # w = None
                # b = None
                # a = None

        ##learner : Adaboost  Nova 1+2 with SVM
                # print("ADA Boost Nova 1+2 with SVM starting...\n")
                # # X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
                # w, b, a = toa.fit(X_train, y_train, M, C,instance_categorization = False, proposed_preprocessing = True,proposed_alpha=True,test_something = True, theta=theta)
                # test_pred7 = toa.predict(X_test, w, b, a, M)
                # precision, recall, fscore, support = score(y_test, test_pred7)
                # S_precision_A12_SVM=S_precision_A12_SVM+precision[1]
                # S_recall_A12_SVM=S_recall_A12_SVM+recall[1]
                # #
                # S_precision_A12_SVM_N=S_precision_A12_SVM_N+precision[0]
                # S_recall_A12_SVM_N=S_recall_A12_SVM_N+recall[0]
                # print("Done.\n")
                # w = None
                # b = None
                # a = None
        ###learner : Adaboost Nova 1+2 with W.SVM 2016 
                print("ADA Boost Nova 1+2 with W.SVM starting...\n")
                w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization = True,proposed_preprocessing= True,proposed_alpha=True,test_something = True, theta=theta)
                test_pred8 = toa.predict(X_test, w, b, a, M)
                precision, recall, fscore, support = score(y_test, test_pred8)
                S_precision_A12_WSVM=S_precision_A12_WSVM+precision[1]
                S_recall_A12_WSVM=S_recall_A12_WSVM+recall[1]
                #
                S_precision_A12_WSVM_N=S_precision_A12_WSVM_N+precision[0]
                S_recall_A12_WSVM_N=S_recall_A12_WSVM_N+recall[0]
                print("Done.\n")

                alpha_test_im12.append(a)
                f2_st_im12="theta="+str(theta)+"_rate_1_"+str(mr)+"TestSize_"+str(i)+"loop"+str(kf)+str(alpha_test_im)+"\n\n\n"
                f2.write(f2_st_im12)

                w = None
                b = None
                a = None
            print(kf)
            print("\n\n FINAL New rate = %s, test size = %s\n\n" %(new_rate,i*0.1))
            # f.write("\n\n Adaboost with SVM:")
            # print("\nAdaboost with SVM:")
            # k_precision_A_SVM='{0:.4f}'.format(S_precision_A_SVM/kf)
            # k_recall_A_SVM='{0:.4f}'.format(S_recall_A_SVM/kf)

            # k_fscore_A_SVM='{0:.4f}'.format((S_precision_A_SVM+S_recall_A_SVM)/(2*kf))
            # #
            # k_precision_A_SVM_N='{0:.4f}'.format(S_precision_A_SVM_N/kf)
            # k_recall_A_SVM_N='{0:.4f}'.format(S_recall_A_SVM_N/kf)
            # k_fscore_A_SVM_N='{0:.4f}'.format((S_precision_A_SVM_N+S_recall_A_SVM_N)/(2*kf))

            # f.write("\nPositive: \n precision \t recall \t fscore \n")
            # f.write(k_precision_A_SVM_N+"\t"+k_recall_A_SVM_N+"\t"+k_fscore_A_SVM_N+"\n")
            # f.write(k_precision_A_SVM+"\t"+k_recall_A_SVM+"\t"+k_fscore_A_SVM+"\n")
            # print("\nPositive: \n precision \t recall \t fscore \n")
            # print(k_precision_A_SVM+"\t"+k_recall_A_SVM+"\t"+k_fscore_A_SVM+"\n")
            # f.write("\n\n\n")

            # f.write("\n\n Adaboost with W.SVM:")
            # print("\nAdaboost with W.SVM:")
            # k_precision_A_WSVM='{0:.4f}'.format(S_precision_A_WSVM/kf)
            # k_recall_A_WSVM='{0:.4f}'.format(S_recall_A_WSVM/kf)
            # k_fscore_A_WSVM='{0:.4f}'.format((S_precision_A_WSVM+S_recall_A_WSVM)/(2*kf))
            # #
            # k_precision_A_WSVM_N='{0:.4f}'.format(S_precision_A_WSVM_N/kf)
            # k_recall_A_WSVM_N='{0:.4f}'.format(S_recall_A_WSVM_N/kf)
            # k_fscore_A_WSVM_N='{0:.4f}'.format((S_precision_A_WSVM_N+S_recall_A_WSVM_N)/(2*kf))
            # #f.write(report(y_test,test_pred).to_string())
            # f.write("\nPositive: \n precision \t recall \t fscore \n")
            # f.write(k_precision_A_WSVM_N+"\t"+k_recall_A_WSVM_N+"\t"+k_fscore_A_WSVM_N+"\n")
            # f.write(k_precision_A_WSVM+"\t"+k_recall_A_WSVM+"\t"+k_fscore_A_WSVM+"\n")
            # print("\nPositive: \n precision \t recall \t fscore \n")
            # print(k_precision_A_WSVM+"\t"+k_recall_A_WSVM+"\t"+k_fscore_A_WSVM+"\n")
            # f.write("\n\n\n")

            # f.write("\n\n Adaboost Nova 1 with SVM:")
            # print("\nAdaboost Nova 1 with SVM:")
            # k_precision_A1_SVM='{0:.4f}'.format(S_precision_A1_SVM/kf)
            # k_recall_A1_SVM='{0:.4f}'.format(S_recall_A1_SVM/kf)
            # k_fscore_A1_SVM='{0:.4f}'.format((S_precision_A1_SVM+S_recall_A1_SVM)/(2*kf))
            # #
            # k_precision_A1_SVM_N='{0:.4f}'.format(S_precision_A1_SVM_N/kf)
            # k_recall_A1_SVM_N='{0:.4f}'.format(S_recall_A1_SVM_N/kf)
            # k_fscore_A1_SVM_N='{0:.4f}'.format((S_precision_A1_SVM_N+S_recall_A1_SVM_N)/(2*kf))

            # f.write("\nPositive: \n precision \t recall \t fscore \n")
            # f.write(k_precision_A1_SVM_N+"\t"+k_recall_A1_SVM_N+"\t"+k_fscore_A1_SVM_N+"\n")
            # f.write(k_precision_A1_SVM+"\t"+k_recall_A1_SVM+"\t"+k_fscore_A1_SVM+"\n")
            # print("\nPositive: \n precision \t recall \t fscore \n")
            # print(k_precision_A1_SVM+"\t"+k_recall_A1_SVM+"\t"+k_fscore_A1_SVM+"\n")
            # f.write("\n\n\n")

            # f.write("\n\n Adaboost Nova 1 with W.SVM:")
            # print("\nAdaboost Nova 1 with W.SVM:")
            # k_precision_A1_WSVM='{0:.4f}'.format(S_precision_A1_WSVM/kf)
            # k_recall_A1_WSVM='{0:.4f}'.format(S_recall_A1_WSVM/kf)
            # k_fscore_A1_WSVM='{0:.4f}'.format((S_precision_A1_WSVM+S_recall_A1_WSVM)/(2*kf))
            # #
            # k_precision_A1_WSVM_N='{0:.4f}'.format(S_precision_A1_WSVM_N/kf)
            # k_recall_A1_WSVM_N='{0:.4f}'.format(S_recall_A1_WSVM_N/kf)
            # k_fscore_A1_WSVM_N='{0:.4f}'.format((S_precision_A1_WSVM_N+S_recall_A1_WSVM_N)/(2*kf))

            # f.write("\nPositive: \n precision \t recall \t fscore \n")
            # f.write(k_precision_A1_WSVM_N+"\t"+k_recall_A1_WSVM_N+"\t"+k_fscore_A1_WSVM_N+"\n")
            # f.write(k_precision_A1_WSVM+"\t"+k_recall_A1_WSVM+"\t"+k_fscore_A1_WSVM+"\n")
            # print("\nPositive: \n precision \t recall \t fscore \n")
            # print(k_precision_A1_WSVM+"\t"+k_recall_A1_WSVM+"\t"+k_fscore_A1_WSVM+"\n")
            # f.write("\n\n\n")

            # f.write("\n\n Adaboost Nova 2 with SVM:")
            # print("\nAdaboost Nova 2 with SVM:")
            # k_precision_A2_SVM='{0:.4f}'.format(S_precision_A2_SVM/kf)
            # k_recall_A2_SVM='{0:.4f}'.format(S_recall_A2_SVM/kf)
            # k_fscore_A2_SVM='{0:.4f}'.format((S_precision_A2_SVM+S_recall_A2_SVM)/(2*kf))
            # #
            # k_precision_A2_SVM_N='{0:.4f}'.format(S_precision_A2_SVM_N/kf)
            # k_recall_A2_SVM_N='{0:.4f}'.format(S_recall_A2_SVM_N/kf)
            # k_fscore_A2_SVM_N='{0:.4f}'.format((S_precision_A2_SVM_N+S_recall_A2_SVM_N)/(2*kf))

            # f.write("\nPositive: \n precision \t recall \t fscore \n")
            # f.write(k_precision_A2_SVM_N+"\t"+k_recall_A2_SVM_N+"\t"+k_fscore_A2_SVM_N+"\n")
            # f.write(k_precision_A2_SVM+"\t"+k_recall_A2_SVM+"\t"+k_fscore_A2_SVM+"\n")
            # print("\nPositive: \n precision \t recall \t fscore \n")
            # print(k_precision_A2_SVM+"\t"+k_recall_A2_SVM+"\t"+k_fscore_A2_SVM+"\n")
            # f.write("\n\n\n")

            # f.write("\n\n Adaboost Nova 2 with W.SVM:")
            # print("\nAdaboost Nova 2 with W.SVM:")
            # k_precision_A2_WSVM='{0:.4f}'.format(S_precision_A2_WSVM/kf)
            # k_recall_A2_WSVM='{0:.4f}'.format(S_recall_A2_WSVM/kf)
            # k_fscore_A2_WSVM='{0:.4f}'.format((S_precision_A2_WSVM+S_recall_A2_WSVM)/(2*kf))
            # #
            # k_precision_A2_WSVM_N='{0:.4f}'.format(S_precision_A2_WSVM_N/kf)
            # k_recall_A2_WSVM_N='{0:.4f}'.format(S_recall_A2_WSVM_N/kf)
            # k_fscore_A2_WSVM_N='{0:.4f}'.format((S_precision_A2_WSVM_N+S_recall_A2_WSVM_N)/(2*kf))

            # f.write("\nPositive: \n precision \t recall \t fscore \n")
            # f.write(k_precision_A2_WSVM_N+"\t"+k_recall_A2_WSVM_N+"\t"+k_fscore_A2_WSVM_N+"\n")
            # f.write(k_precision_A2_WSVM+"\t"+k_recall_A2_WSVM+"\t"+k_fscore_A2_WSVM+"\n")
            # print("\nPositive: \n precision \t recall \t fscore \n")
            # print(k_precision_A2_WSVM+"\t"+k_recall_A2_WSVM+"\t"+k_fscore_A2_WSVM+"\n")
            # f.write("\n\n\n")

            # f.write("\n\n Adaboost Nova 1+2 with SVM:")
            # print("\nAdaboost Nova 1+2 with SVM:")
            # k_precision_A12_SVM='{0:.4f}'.format(S_precision_A12_SVM/kf)
            # k_recall_A12_SVM='{0:.4f}'.format(S_recall_A12_SVM/kf)
            # k_fscore_A12_SVM='{0:.4f}'.format((S_precision_A12_SVM+S_recall_A12_SVM)/(2*kf))
            # #
            # k_precision_A12_SVM_N='{0:.4f}'.format(S_precision_A12_SVM_N/kf)
            # k_recall_A12_SVM_N='{0:.4f}'.format(S_recall_A12_SVM_N/kf)
            # k_fscore_A12_SVM_N='{0:.4f}'.format((S_precision_A12_SVM_N+S_recall_A12_SVM_N)/(2*kf))

            # f.write("\nPositive: \n precision \t recall \t fscore \n")
            # f.write(k_precision_A12_SVM_N+"\t"+k_recall_A12_SVM_N+"\t"+k_fscore_A12_SVM_N+"\n")
            # f.write(k_precision_A12_SVM+"\t"+k_recall_A12_SVM+"\t"+k_fscore_A12_SVM+"\n")
            # print("\nPositive: \n precision \t recall \t fscore \n")
            # print(k_precision_A12_SVM+"\t"+k_recall_A12_SVM+"\t"+k_fscore_A12_SVM+"\n")
            # f.write("\n\n\n")

            # f.write("\n\n Adaboost Nova 1+2 with W.SVM:")
            # print("\nAdaboost Nova 1+2 with W.SVM:")
            # k_precision_A12_WSVM='{0:.4f}'.format(S_precision_A12_WSVM/kf)
            # k_recall_A12_WSVM='{0:.4f}'.format(S_recall_A12_WSVM/kf)
            # k_fscore_A12_WSVM='{0:.4f}'.format((S_precision_A12_WSVM+S_recall_A12_WSVM)/(2*kf))
            # #
            # k_precision_A12_WSVM_N='{0:.4f}'.format(S_precision_A12_WSVM_N/kf)
            # k_recall_A12_WSVM_N='{0:.4f}'.format(S_recall_A12_WSVM_N/kf)
            # k_fscore_A12_WSVM_N='{0:.4f}'.format((S_precision_A12_WSVM_N+S_recall_A12_WSVM_N)/(2*kf))
            # #f.write(report(y_test,test_pred).to_string())
            # f.write("\nPositive: \n precision \t recall \t fscore \n")
            # f.write(k_precision_A12_WSVM_N+"\t"+k_recall_A12_WSVM_N+"\t"+k_fscore_A12_WSVM_N+"\n")
            # f.write(k_precision_A12_WSVM+"\t"+k_recall_A12_WSVM+"\t"+k_fscore_A12_WSVM+"\n")
            # print("\nPositive: \n precision \t recall \t fscore \n")
            # print(k_precision_A12_WSVM+"\t"+k_recall_A12_WSVM+"\t"+k_fscore_A12_WSVM+"\n")
            # f.write("\n\n\n")

        


  
        #learner : Adaboost
    # f.write("\n\n Adaboost with SVM (scratch):")
    # for i in range(3, 4):
    #     X_train, y_train, X_test, y_test = Vertebral_column.load_data(test_size= i * 0.1, new_rate=new_rate)
    #     f.write("\n\n New rate = %s, test size = %s, C = %s, M = %s\n\n" %(new_rate,i*0.1, C, M))
    #     w, b, a = toa.fit(X_train, y_train, M, C, instance_categorization=False, proposed_preprocessing = False,test_something = True)
    #     test_pred = toa.predict(X_test, w, b, a, M)
    #     f.write(report(y_test,test_pred).to_string())
    #     f.write("\n\n")
    #     f.write(roc_auc_score(y_test, test_pred).to_string())
    #     f.write("\n\n\n")

    f.close()
    f2.close()
