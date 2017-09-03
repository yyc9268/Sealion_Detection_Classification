import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib

##define base model
def base_model():
     model = Sequential()
     model.add(Dense(14, input_dim=4, init='normal', activation='relu'))
     model.add(Dense(7, init='normal', activation='relu'))
     model.add(Dense(1, init='normal'))
     model.compile(loss='mean_squared_error', optimizer = 'adam')
     return model

def pred_pups(r_test):
    data = np.genfromtxt('input/train.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5))
    #real_data = np.genfromtxt('input/real.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5))

    X_train = data[0:750, :4]
    Y_train = data[0:750, 4]
    Y_test = data[750:948, 4]

    new_input = []
    new_gt = []

    for i in range(len(r_test)):
        if r_test[i] != False:
            new_gt.append(Y_test[i])
            new_input.append(r_test[i])

    seed = 2
    np.random.seed(seed)
    print("X_train: ", len(X_train))
    print("Y_test : ", len(Y_test))
    print("r_test : ", len(r_test))

    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(new_input)

    clf = KerasRegressor(build_fn=base_model, nb_epoch=500, batch_size=5,verbose=0)

    clf.fit(X_train,Y_train)
    result = clf.predict(X_test)
    """
    plt.scatter(new_gt, result, label='regression model')
    plt.plot(np.arange(np.max(Y_test)), np.arange(np.max(Y_test)), color='k', label='perfect prediction')
    plt.title('predictions of the last model')
    plt.legend(loc='best')
    plt.xlabel('true #pups')
    plt.ylabel('predicted #pups')
    plt.show()
    """
    return result, new_input

def final_pred(r_test):
    data = np.genfromtxt('input/train.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5))
    #real_data = np.genfromtxt('input/real.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5))

    X_train = data[0:750, :4]
    Y_train = data[0:750, 4]

    seed = 2
    np.random.seed(seed)

    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.fit_transform(r_test)

    clf = KerasRegressor(build_fn=base_model, nb_epoch=500, batch_size=5,verbose=0)

    clf.fit(X_train,Y_train)
    result = clf.predict(X_test)

    return result


def xgb_pups(r_test):
    data = np.genfromtxt('input/train.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5))
    # real_data = np.genfromtxt('input/real.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5))
    X_train = data[0:750, :4]
    Y_train = data[0:750, 4]
    Y_test = data[750:800, 4]

    new_input = []
    new_gt = []
    for i in range(len(r_test)):
        if r_test[i] != False:
            new_gt.append(Y_test[i])
            new_input.append(r_test[i])

    #RMSE = np.zeros(n_sims)
    #f_imp = np.zeros([n_sims, np.shape(X)[1]])
    for i in range(1):
        # split the data
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        # initialize XGBRegressor
        GB = xgb.XGBRegressor()

        # the parameter grid below was too much on the kaggle kernel
        #param_grid = {"learning_rate": [0.01,0.03,0.1],
        #              "objective": ['reg:linear'],
        #              "n_estimators": [300,1000,3000]}
        # do GridSearch
        #search_GB = GridSearchCV(GB,param_grid,cv=4,n_jobs=-1).fit(X_train,Y_train)
        # the best parameters should not be on the edges of the parameter grid
        # print('   ',search_GB.best_params_)
        # train the best model
        #xgb_pups = xgb.XGBRegressor(**search_GB.best_params_).fit(X_train, Y_train)

        # preselected parameters
        param_grid = {"learning_rate": 0.03,
                      "objective": 'reg:linear',
                      "n_estimators": 300}
        xgb_pups = xgb.XGBRegressor(**param_grid).fit(X_train, Y_train)

        # predict on the test set
        preds = xgb_pups.predict(new_input)

    # visualize the prediction of the last model
    plt.scatter(new_gt, preds, label='regression model')
    plt.plot(np.arange(np.max(Y_test)), np.arange(np.max(Y_test)), color='k', label='perfect prediction')
    plt.title('predictions of the last model')
    plt.legend(loc='best')
    plt.xlabel('true #pups')
    plt.ylabel('predicted #pups')
    plt.show()

    return preds, new_input

def xgb_tune():
    data = np.genfromtxt('input/info/info.csv', delimiter=',', skip_header=1, usecols=(0, 1, 2, 3))
    # real_data = np.genfromtxt('input/real.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5))
    X_train = data[0:100, 0:3]
    Y_train = data[0:100, 3]
    Y_test = data[100:126, 3]
    new_input = data[100:126, 0:3]

    #new_input = []
    #new_gt = []
    """
    for i in range(len(r_test)):
        if r_test[i] != False:
            new_gt.append(Y_test[i])
            new_input.append(r_test[i])
    """
    #RMSE = np.zeros(n_sims)
    #f_imp = np.zeros([n_sims, np.shape(X)[1]])
    for i in range(1):
        # split the data
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        # initialize XGBRegressor
        GB = xgb.XGBRegressor()

        # the parameter grid below was too much on the kaggle kernel
        #param_grid = {"learning_rate": [0.01,0.03,0.1],
        #              "objective": ['reg:linear'],
        #              "n_estimators": [300,1000,3000]}
        # do GridSearch
        #search_GB = GridSearchCV(GB,param_grid,cv=4,n_jobs=-1).fit(X_train,Y_train)
        # the best parameters should not be on the edges of the parameter grid
        # print('   ',search_GB.best_params_)
        # train the best model
        #xgb_pups = xgb.XGBRegressor(**search_GB.best_params_).fit(X_train, Y_train)

        # preselected parameters
        param_grid = {"learning_rate": 0.03,
                      "objective": 'reg:linear',
                      "n_estimators": 300}
        xgb_pups = xgb.XGBRegressor(**param_grid).fit(X_train, Y_train)

        # predict on the test set
        preds = xgb_pups.predict(new_input)

    # visualize the prediction of the last model
    plt.scatter(Y_test, preds, label='regression model')
    plt.plot(np.arange(np.max(Y_test)), np.arange(np.max(Y_test)), color='k', label='perfect prediction')
    plt.title('predictions of the last model')
    plt.legend(loc='best')
    plt.xlabel('true #pups')
    plt.ylabel('predicted #pups')
    plt.show()

    return preds, new_input

#f_names = ['adult males','subadult males','adult females','juveniles']

if __name__ == "__main__":
    #data = np.genfromtxt('input/train.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5))
    #real_data = np.genfromtxt('input/real.csv', delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5))

    #r_test = data[0, :4]
    xgb_tune()

    #pup_num = pred_pups(r_test)