import random, psutil
import matplotlib.pyplot as plt
import pandas as pd

def dataset():
    random.seed(int(psutil.cpu_percent(1)*psutil.virtual_memory()[3]%psutil.virtual_memory()[4]))

    print('Loading data....')
    data = pd.read_excel("LED_dataset01.xls",skiprows=6)
    print('Loaded ',len(data),' rows and ',len(data.columns),' columns\n\n' )
    print(data.head(5))

    print(data.columns)

    data['date']=pd.to_datetime(data['Местное время в Пулково (аэропорт)'],dayfirst=True)
    x=data['date']
    y=data['T']

    plt.plot(x,y)
    plt.show()

    # shorten
    condition = (data['date'].between('2017-11-01','2018-03-01'))
    data_short = data[condition]
    plt.plot(data_short['date'],data_short['T'])
    plt.show()



if __name__=="__main__":
    dataset()