import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# wind speed table for conversions
wind_rus = ["Штиль, безветрие","Ветер, дующий с севера","Ветер, дующий с северо-северо-востока","Ветер, дующий с северо-востока","Ветер, дующий с востоко-северо-востока","Ветер, дующий с востока","Ветер, дующий с востоко-юго-востока","Ветер, дующий с юго-востока","Ветер, дующий с юго-юго-востока","Ветер, дующий с юга","Ветер, дующий с юго-юго-запада","Ветер, дующий с юго-запада","Ветер, дующий с западо-юго-запада","Ветер, дующий с запада","Ветер, дующий с западо-северо-запада","Ветер, дующий с северо-запада","Ветер, дующий с северо-северо-запада","Переменное направление"]
wind_naut = ["STILL","N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW","VARIABLE"]
wind_deg = [-1,0,22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5,360]


def dataset():

    print('Loading data....')
    data = pd.read_excel("LED_dataset01.xls",skiprows=6)
    # print(data) # unnecessary anymore
    
    data = data[data['T'].notna()]
    print('Loaded ',len(data),' rows and ',len(data.columns),' columns\n\n' )

    # 
    # Data columns:
    # T - temperature
    # P0 - pressure at the station height
    # P - pressure normalized to sea level
    # U -  relative humidity in %
    # DD - wind direction in Russian
    # 
    needed_cols = set(['Местное время в Пулково (аэропорт)','T','P0','P','U','DD'])

    print('Prepare data...') 

    #remove unnecessary columns
    dfcols = data.columns.tolist() 
    for i in dfcols:
        if i not in needed_cols:
            data = data.drop([i],axis=1) 

    data['date']=pd.to_datetime(data['Местное время в Пулково (аэропорт)'],dayfirst=True)
    data['dayofyear'] = data['date'].dt.dayofyear

    # wind direction conversions
    wind_replace = dict(zip(wind_rus, wind_deg))
    data = data.replace({"DD":wind_replace})

    print(data.head(5))

    me_test, me_train = LinearRegr(data,'T',False)
    print("========= Linear Regression =============")
    print('Mean error on train set: ',me_train)    
    print('Mean error on test set: ', me_test)    

    me_test, me_train = DecisionTreeRegr(data,"T",False)
    print("========= Decision Tree Regression =============")
    print('Mean error on train set: ',me_train)    
    print('Mean error on test set: ', me_test)  

    me_test, me_train = LinearRegr2(data,'T',False)
    print("========= Linear Regression with 2 columns: cos_dayofyear and DD =============")
    print('Mean error on train set: ',me_train)    
    print('Mean error on test set: ', me_test)    

    me_test, me_train = DecisionTreeRegr2(data,"T",False)
    print("========= Decision Tree Regression with 2 columns: cos_dayofyear and DD =============")
    print('Mean error on train set: ',me_train)    
    print('Mean error on test set: ', me_test)  

    # For humidity (U) worse:
    # ========= Linear Regression =============
    # Mean error on train set:  11.058175651102943
    # Mean error on test set:  11.330622966316202
    # ========= Decision Tree Regression =============
    # Mean error on train set:  10.482040816951246
    # Mean error on test set:  11.202803224126097 
    # 
    # For Wind direction absolutely unfit:
    # ========= Linear Regression =============
    # Mean error on train set:  81.14177944367745
    # Mean error on test set:  83.48014131340648
    # ========= Decision Tree Regression =============
    # Mean error on train set:  77.23867671954586
    # Mean error on test set:  84.4171004566516 
    # ========= Linear Regression =============
    # Mean error on train set:  3.940361456601304
    # Mean error on test set:  4.5044020624159415
    # ========= Decision Tree Regression (max_depth=3) =============
    # Mean error on train set:  3.478650003941484
    # Mean error on test set:  4.63285184688049
    # ========= Linear Regression with 2 columns: cos_dayofyear and DD =============
    # Mean error on train set:  3.9404235986175467
    # Mean error on test set:  4.505042685389038
    # ========= Decision Tree Regression with 2 columns: cos_dayofyear and DD (max_depth=3)  =============
    # Mean error on train set:  3.478650003941484
    # Mean error on test set:  4.63285184688049    

    # ========= Decision Tree Regression (max_depth=5) =============
    # Mean error on train set:  3.0437196280041023
    # Mean error on test set:  4.608336451857637
    # ========= Decision Tree Regression with 2 columns: cos_dayofyear and DD (max_depth=5) =============
    # Mean error on train set:  3.005452977825302
    # Mean error on test set:  4.499927369539333   
    # 
    # ========= Decision Tree Regression (max_depth=8)  =============
    # Mean error on train set:  2.7130763905262603
    # Mean error on test set:  4.93952034674399
    # ========= Decision Tree Regression with 2 columns: cos_dayofyear and DD (max_depth=8)  =============
    # Mean error on train set:  2.469992081346145
    # Mean error on test set:  4.774331831074269 


def DecisionTreeRegr(data, Y_col='T',plotit=True):
    data_train = data[data['date'] < '2020-01-01']
    data_test = data[data['date'] >= '2020-01-01']

    X_train = pd.DataFrame()
    X_train['dayofyear'] = data_train['dayofyear']

    X_test = pd.DataFrame()
    X_test['dayofyear']=data_test['dayofyear']

    y_train = data_train[Y_col]
    y_test = data_test[Y_col]

    model = DecisionTreeRegressor(max_depth=8,min_samples_split=2)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    if plotit:
        plt.figure(figsize=(20,5))
        plt.scatter(data_train['date'],data_train[Y_col], label='Train',color='red')
        plt.scatter(data_test['date'],data_test[Y_col],label='Test',color='blue')
        plt.scatter(data_train['date'],pred_train, label='Pred Train',color='yellow')
        plt.scatter(data_test['date'],pred_test,label='Pred Test',color='green')
        plt.legend()
        plt.show()

    return mean_absolute_error(y_test, pred_test), mean_absolute_error(y_train, pred_train)    

def DecisionTreeRegr2(data, Y_col='T',plotit=True):
    data_train = data[data['date'] < '2020-01-01']
    data_test = data[data['date'] >= '2020-01-01']

    X_train = pd.DataFrame()
    X_train['dayofyear'] = data_train['dayofyear']
    X_train['DD'] = data_train['DD']

    X_test = pd.DataFrame()
    X_test['dayofyear']=data_test['dayofyear']
    X_test['DD']=data_test['DD']

    y_train = data_train[Y_col]
    y_test = data_test[Y_col]

    model = DecisionTreeRegressor(max_depth=8,min_samples_split=2)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    if plotit:
        plt.figure(figsize=(20,5))
        plt.scatter(data_train['date'],data_train[Y_col], label='Train',color='red')
        plt.scatter(data_test['date'],data_test[Y_col],label='Test',color='blue')
        plt.scatter(data_train['date'],pred_train, label='Pred Train',color='yellow')
        plt.scatter(data_test['date'],pred_test,label='Pred Test',color='green')
        plt.legend()
        plt.show()

    return mean_absolute_error(y_test, pred_test), mean_absolute_error(y_train, pred_train)    


def LinearRegr(data, Y_col='T',plotit=True):
    scaled_dayofyear = ((data['dayofyear']-1)/366)*2*np.pi
    data['cos_dayofyear'] = np.sin(scaled_dayofyear-np.pi/2-np.pi/9.9)
    
    data_train = data[data['date'] < '2020-01-01']
    data_test = data[data['date'] >= '2020-01-01']

    X_train = pd.DataFrame()
    X_train['cos_dayofyear'] = data_train['cos_dayofyear']

    X_test = pd.DataFrame()
    X_test['cos_dayofyear']=data_test['cos_dayofyear']

    y_train = data_train[Y_col]
    y_test = data_test[Y_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    if plotit:
        plt.figure(figsize=(20,5))
        plt.scatter(data_train['date'],data_train[Y_col], label='Train',color='red')
        plt.scatter(data_test['date'],data_test[Y_col],label='Test',color='blue')
        plt.scatter(data_train['date'],pred_train, label='Pred Train',color='yellow')
        plt.scatter(data_test['date'],pred_test,label='Pred Test',color='green')
        plt.legend()
        plt.show()

    return mean_absolute_error(y_test, pred_test), mean_absolute_error(y_train, pred_train)    


def LinearRegr2(data, Y_col='T',plotit=True):
    scaled_dayofyear = ((data['dayofyear']-1)/366)*2*np.pi
    data['cos_dayofyear'] = np.sin(scaled_dayofyear-np.pi/2)
    
    data_train = data[data['date'] < '2020-01-01']
    data_test = data[data['date'] >= '2020-01-01']

    X_train = pd.DataFrame()
    X_train['cos_dayofyear'] = data_train['cos_dayofyear']
    X_train['DD'] = data_train['DD']

    X_test = pd.DataFrame()
    X_test['cos_dayofyear']=data_test['cos_dayofyear']
    X_test['DD']=data_test['DD']

    y_train = data_train[Y_col]
    y_test = data_test[Y_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    if plotit:
        plt.figure(figsize=(20,5))
        plt.scatter(data_train['date'],data_train[Y_col], label='Train',color='red')
        plt.scatter(data_test['date'],data_test[Y_col],label='Test',color='blue')
        plt.scatter(data_train['date'],pred_train, label='Pred Train',color='yellow')
        plt.scatter(data_test['date'],pred_test,label='Pred Test',color='green')
        plt.legend()
        plt.show()

    return mean_absolute_error(y_test, pred_test), mean_absolute_error(y_train, pred_train)    


if __name__=="__main__":
    dataset()