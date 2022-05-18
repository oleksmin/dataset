import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    wind_replace = dict(zip(wind_rus, wind_naut))
    data = data.replace({"DD":wind_replace})

    print(data.head(5))

    # if not data.empty: return 1 # unnecessary anymore
    
    scaled_dayofyear = ((data['dayofyear']-1)/366)*2*np.pi
    data['cos_dayofyear'] = np.sin(scaled_dayofyear-np.pi/2) # to make min values on Jan 1 and max values on Jul 1
    
    data_train = data[data['date'] < '2020-01-01']
    data_test = data[data['date'] >= '2020-01-01']

    X_train = pd.DataFrame()
    X_train['dayofyear'] = data_train['dayofyear']

    X_test = pd.DataFrame()
    X_test['dayofyear']=data_test['dayofyear']

    y_train = data_train['T']
    y_test = data_test['T']


    model = DecisionTreeRegressor(max_depth=3,min_samples_split=2)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    plt.figure(figsize=(20,5))
    plt.plot(data_train['date'],data_train['T'], label='Train',color='red')
    plt.plot(data_test['date'],data_test['T'],label='Test',color='blue')
    plt.plot(data_train['date'],pred_train, label='Pred Train',color='yellow')
    plt.plot(data_test['date'],pred_test,label='Pred Test',color='green')
    plt.legend()
    plt.show()

    print('Mean error on train set: ', mean_absolute_error(y_train, pred_train))    
    print('Mean error on test set: ', mean_absolute_error(y_test, pred_test))    


if __name__=="__main__":
    dataset()