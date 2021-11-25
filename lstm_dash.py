import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import statsmodels.api as sm
import dash
import dash_bootstrap_components as dbc

from dash import html
from dash import dcc
from dash import dash_table
from dash.dependencies import Input, Output, State
from pandas_datareader import data
from matplotlib.ticker import FuncFormatter 
from scipy.stats import norm
from numpy import array
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from datetime import datetime
from datetime import timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = dbc.Container(
    [
     html.Div(id="testing-lstm"),
     dbc.Button("test lstm", id="test_button", n_clicks=0)
    ]
)

@app.callback(
    Output("testing-lstm", "children"),
    Input("test_button", "n_clicks"),
    prevent_initial_call = True
)
def testing_lstm(clicks):
    tickers_list = ['XWC-USD', 'DGD-USD', 'ETH-USD', 'BTC-USD', 'GNO-USD', 'LRC-USD', 'XEM-USD', 'ANT-USD', 'ADA-USD', 'LINK-USD']

    START_DATE = '2020-01-01'
    END_DATE = '2021-12-31'


    prices_df = yf.download(tickers_list, start=START_DATE, end=END_DATE, adjusted=True)
    prices_df = prices_df['Close']
    results = []

    original_price = []
    for ticker in tickers_list:
        current_price = prices_df[ticker].iloc[-1]
        original_price.append(current_price)
    original_price
    dataFrames_arr = []

    for ticker in tickers_list:

        close_df = prices_df[ticker]


        scaler=MinMaxScaler(feature_range=(0,1))
        close_df=scaler.fit_transform(np.array(close_df).reshape(-1,1))

        training_size=int(len(close_df)*0.65)
        test_size=len(close_df)-training_size
        train_data,test_data=close_df[0:training_size,:],close_df[training_size:len(close_df),:1]


        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)

        X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

        model= Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam') 
        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=1,batch_size = 32,verbose =1)

        shape_input = test_data.shape[0]-100


        x_input=test_data[shape_input:].reshape(1,-1)

        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()



        lst_output=[]
        n_steps=100
        i=0
        while(i<30):
            
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                # print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                # print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                # print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                # print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1


        df3=close_df.tolist()
        df3.extend(lst_output)

        df3=scaler.inverse_transform(df3).tolist()
        dataFrames_arr.append(df3)

        results.append(df3[-1])

    future_prices = []

    for pr in results:
        future_prices.append(pr[0])

    array = []

    for i in range(len(original_price)):
        eachTicker = []
        eachTicker.append(original_price[i])
        eachTicker.append(future_prices[i])
        array.append(eachTicker)


    time_extension = len(df3) 

    EndDate = prices_df.index[0] + timedelta(days=time_extension)

    dateIndexes = pd.date_range(start='2020-01-01', end=EndDate)

    df = pd.DataFrame(data = array, 
                      index = tickers_list, 
                      columns = ['original_prices','future_prices'])
    df['pct_change'] = ((df['future_prices'] - df['original_prices'])/df['original_prices']) * 100
    df.sort_values(by='pct_change',ascending = False,inplace = True)

    new_df = df.copy()
    new_df['weights'] = [24, 21, 15, 13, 9, 7, 5, 3, 2 , 1 ]

    name = tickers_list[0]
    column_name = name + " Price"
    each_df = pd.DataFrame(data = dataFrames_arr[0], 
                      index = dateIndexes, 
                      columns = [column_name])
      


    for j in range(1, len(dataFrames_arr)):
        name = tickers_list[j]
        column_name = name + " Price"
        each_df2 = pd.DataFrame(data = dataFrames_arr[j], 
                        index = dateIndexes, 
                        columns = [column_name])
        each_df = pd.merge(each_df, each_df2, left_index=True, right_index=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for k in range(len(tickers_list)):
        name = tickers_list[k]
        column_name = name + " Price"

        # Add traces
        fig.add_trace(
            go.Scatter(x=dateIndexes, y=each_df[column_name], name=column_name),
            secondary_y=False,
        )

    fig.show()
    new_df

    figure = dcc.Graph(figure = fig)
    # print(new_df)
    new_df = new_df.reset_index()
    table = dbc.Table.from_dataframe(new_df, striped = True , bordered = True, hover = True)

    return [figure, table]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
        
    return numpy.array(dataX), numpy.array(dataY)
  
if __name__ == "__main__":
  app.run_server(debug=True, use_reloader=False)