import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import sklearn
from sklearn.linear_model import LinearRegression
# print('success')

def mlr_test(string="empty"):
    """
        mlr_test
        Arguments:
            string <string> -- the string that is printed to the user
    """
    print(string)

def compareMlrResults(col_arr, x_train,x_valid,y_train,y_valid,p=True):
    """
        Description: 
            Find the best MLR model through taking out unique combonations of column names provided by col_arr. 
        Args:
            col_arr <list> -- list of column names that could be removed to make the mlr model better in some way (r2, mae, rmse)
            x_train <pandas df> -- traing x vars
            x_valid <pandas df> -- valid x vars
            y_train <pandas df> -- train y outcome
            y_valid <pandas df> -- valid y outcome 
            p <bool> -- print details (default true) 
    """
    r2 = None
    mae = None
    rmse = None
    for L in range(0, len(col_arr)+1):
        for subset in itertools.combinations(col_arr, L):
            remove_col = list(subset)
            result = runMlr(y_train,y_valid,x_train,x_valid,remove_col,)
            if r2 == None:
                r2 = result['r2']
                mae = result['mae']
                rmse = result['rmse']
                best_r2 = result
                best_mae = result
                best_rmse = result
                r2_col = remove_col
                mae_col = remove_col
                rmse_col = remove_col
            else:
                if result['r2'] > r2:
                    best_r2 = result
                    r2_col = remove_col
                if result['mae'] < mae:
                    best_mae = result
                    mae_col = remove_col
                if result['rmse'] < rmse:
                    best_rmse = result
                    rmse_col = remove_col
                
    #             print(list(subset))
    if p:
        print('---------------result---------------')
        print('best r2: ',best_r2,'\n removed cols', r2_col)
        print('best mae: ',best_mae,'\n removed cols',mae_col)
        print('best rsme: ',best_rmse,'\n removed cols',rmse_col)
    return r2_col

def predict(x_valid, y_valid, mlr):
    """
        Desc:
        runs a trained mlr model on validation data and returns the resulting r2, mae, rmse 

        Args:
        x_valid <pandas df> -- valid x vars
        y_valid <pandas df> -- valid y outcome
        mlr <object> -- from sklearn, this is the mlr model it has already been fitted (trained)

        return <object> -- {r2, mae, rmse}
    """
    print('----------------Validate--------------')
    # create y predict from our mlr model
    y_pred = mlr.predict(x_valid)  
    R2 = sklearn.metrics.r2_score(y_valid, y_pred)
    MAE = sklearn.metrics.mean_absolute_error(y_valid, y_pred)
    RMSE = np.sqrt(sklearn.metrics.mean_squared_error(y_valid, y_pred))
    
    print('Model quality results:  R2:', format(R2, '0.3f'),'  MAE:', format(MAE, '0.1f'), '  RMSE:', format(RMSE,'0.1f'))
    return {'r2':R2, 'mae':MAE, 'rmse':RMSE}

def buildMlr(x_train,x_valid,y_train,rm_col_arr,show_summary=False):
    """
        Description: 
            builds and trains an MLR model. 
        Args:
            x_train <pandas df> -- traing x vars
            x_valid <pandas df> -- valid x vars
            y_train <pandas df> -- train y outcome
            rm_col_arr <list> -- list of column names that could be removed to make the mlr model better in some way (r2, mae, rmse)
            show_summary <bool> -- print details (default Fales) 
        Return:
        mlr <object> -- sklearn trained, mlr model
    """
    # create mlr model
    mlr = LinearRegression()
    #Now we fit (train) the model using the training data. This model got trained from 
    #the DataFrame that contains all time variables and dummy variables.
    # train model
    mlr.fit(x_train, y_train)
    X2 = sm.add_constant(x_train)
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    if show_summary:
        print(est2.summary())
    return mlr

def runMlr(y_train,y_valid,x_train, x_valid, rm_col_arr, show_summary=False):
    """
        Description: 
            builds and trains an MLR model. 
        Args:
            y_train <pandas df> -- train y outcome
            y_valid <pandas df> -- valid y outcome
            x_train <pandas df> -- traing x vars
            x_valid <pandas df> -- valid x vars
            rm_col_arr <list> -- list of column names that could be removed to make the mlr model better in some way (r2, mae, rmse)
            show_summary <bool> -- print details (default Fales) 
        Return:
        mlr <object> -- sklearn trained, mlr model
    """
    _x_train = x_train.drop(columns=rm_col_arr)
    _x_valid = x_valid.drop(columns=rm_col_arr)
    mlr = buildMlr(_x_train,_x_valid,y_train,show_summary,rm_col_arr)
    print('results after taking out these columns: ')
    print(' '.join(map(str, rm_col_arr)))
    return predict(_x_valid, y_valid, mlr)

def mlrForTimeSeries(y_train_valid,x_train_valid,x_forecast,arr_col_rmv,show_osl=True):
    """
        Description: 
            builds an mlr forcast form y_train_valid and x_train_valid. 
        Args:
            y_train_valid <pandas df> -- train y outcome
            x_train_valid <pandas df> -- traing x vars
            x_forecast <pandas df> -- x variables we will use to make the forcast
            arr_col_rmv <list> -- list of column names that could be removed to make the mlr model better in some way (r2, mae, rmse)
            show_osl <bool> -- print details (default Fales) 
        Return:
        y_pred <list> -- results from the trained model
    """
    x_train_valid = x_train_valid.drop(columns=arr_col_rmv)
    x_forecast = x_forecast.drop(columns=arr_col_rmv)
    mlr = buildMlr(x_train_valid,x_forecast,y_train_valid,show_osl,arr_col_rmv)
    print("\n\n----------------------------------------------")
    y_pred = mlr.predict(x_train_valid)  
    R2 = sklearn.metrics.r2_score(y_train_valid, y_pred)
    MAE = sklearn.metrics.mean_absolute_error(y_train_valid, y_pred)
    RMSE = np.sqrt(sklearn.metrics.mean_squared_error(y_train_valid, y_pred))
     
    print('Model quality results:  R2:', format(R2, '0.3f'),'  MAE:', format(MAE, '0.1f'), '  RMSE:', format(RMSE,'0.1f'))
    return y_pred

def timeSeriesForecastY(x_train_valid,x_forecast,y_train_valid,arr_col_rmv):
    """
        Description: 
            builds an mlr forcast form y_train_valid and x_train_valid. 
        Args:
            x_train_valid <pandas df> -- traing x vars
            x_forecast <pandas df> -- x variables we will use to make the forcast
            y_train_valid <pandas df> -- train y outcome
            arr_col_rmv <list> -- list of column names that could be removed to make the mlr model better in some way (r2, mae, rmse)
        Return:
        y_pred <list> -- results from the trained model
    """
    mlr = LinearRegression()
    mlr.fit(x_train_valid.drop(columns=arr_col_rmv), y_train_valid)
    my_forcast = mlr.predict(x_forecast.drop(columns=arr_col_rmv))
    return my_forcast

def help(func_name=None):
    functions={
        'runMlr':['y_train','y_valid','x_train', 'x_valid', 'rm_col_arr', 'show_summary'],
        'compareMlrResults':['col_arr', 'x_train','x_valid','x_train_valid','y_train','y_valid','p'],
        'predict':['x_valid', 'y_valid', 'mlr'],
        'help':['func_name'],
    }
    if func_name==None:
        for key in functions:
            print(key, functions[key])
    else:
        print(func_name,functions[func_name])
