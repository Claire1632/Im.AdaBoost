import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from common.change_rate_data import change_rate_data

def load_data(test_size):
    dataset = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/LEC7/madaboost_23102022/madaboost/data/datasets/abalone.csv')
    dataset_desc = dataset.describe(include = 'all')
    abalone_map = {15 : 1, 1 : -1, 2 : -1, 3 : -1, 4 : -1, 5 : -1, 6 : -1, 7 : -1, 8 : -1, 9 : -1, 10 : -1,
    11 : -1, 12 : -1, 13 : -1, 14 : -1, 16 : -1, 17 : -1, 18 : -1, 19 : -1, 20 : -1, 21 : -1, 22 : -1, 23 : -1,
    24 : -1, 25 : -1, 26 : -1, 27 : -1, 28 : -1, 29 : -1}
    dataset['Rings'] = dataset['Rings'].map(abalone_map)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    labelencoder_X = LabelEncoder()
    # Encoding the Sex Categorization
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    onehotencoder_X = OneHotEncoder(handle_unknown='ignore')
    onehotencoder_X.fit_transform(X).toarray()

    #Split data
    #X, y = change_rate_data(X, y , new_rate = new_rate)
    X_train, X_test, y_train, y_test = tts(X, y, test_size = test_size, random_state = 42, stratify=y)
    #Scalling Data
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    
    return X_train, y_train, X_test, y_test