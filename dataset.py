import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def dataset():

    print('Loading data....')
    data = pd.read_excel("LED_dataset01.xls",skiprows=6)
    data = data[data['T'].notna()]
    print('Loaded ',len(data),' rows and ',len(data.columns),' columns\n\n' )

    
    data['date']=pd.to_datetime(data['Местное время в Пулково (аэропорт)'],dayfirst=True)
    
    data['dayofyear'] = data['date'].dt.dayofyear

    scaled_dayofyear = ((data['dayofyear']-1)/366)*2*np.pi
    data['cos_dayofyear'] = np.sin(scaled_dayofyear-np.pi/2)
    
    data_train = data[data['date'] < '2020-01-01']
    data_test = data[data['date'] >= '2020-01-01']

    X_train = pd.DataFrame()
    X_train['cos_dayofyear'] = data_train['cos_dayofyear']

    X_test = pd.DataFrame()
    X_test['cos_dayofyear']=data_test['cos_dayofyear']

    y_train = data_train['T']
    y_test = data_test['T']


    model = LinearRegression()
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

    print('Mean error on test set: ', mean_absolute_error(y_test, pred_test))    


if __name__=="__main__":
    dataset()