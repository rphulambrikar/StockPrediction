import streamlit as st
import pandas_datareader as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from numpy import array

st.title('Stock Trend Prediction of Apple Stocks')


initial_df = pdr.get_data_tiingo('AAPL', api_key='777188ec604cc433ed7cc1d272f7d01392053130')

initial_df.to_csv('AAPL.csv')


initial_df=pd.read_csv('AAPL.csv')



#describing data
st.subheader('Data from last 5 years(Dec 2016 to Dec 2021)')
st.write(initial_df.describe())

final_df = initial_df.reset_index()['close']

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(final_df)
st.pyplot(fig)


scaler = MinMaxScaler(feature_range=(0,1))
final_df = scaler.fit_transform(np.array(final_df).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(final_df)*0.65)
test_size=len(final_df)-training_size
train_data,test_data=final_df[0:training_size,:],final_df[training_size:len(final_df),:1]


def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = load_model('stacked_lstm.h5')

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

math.sqrt(mean_squared_error(ytest,test_predict))

look_back=100
trainPredictPlot = np.empty_like(final_df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(final_df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(final_df)-1, :] = test_predict
# plot baseline and predictions
fig2 = plt.figure(figsize = (12,6))
plt.plot(scaler.inverse_transform(final_df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
st.subheader('Spliting Training and Testing Data')
st.pyplot(fig2)

x_input=test_data[341:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output = []
n_steps = 100
i = 0
while (i < 30):

	if (len(temp_input) > 100):
		# print(temp_input)
		x_input = np.array(temp_input[1:])
		x_input = x_input.reshape(1, -1)
		x_input = x_input.reshape((1, n_steps, 1))
		# print(x_input)
		yhat = model.predict(x_input, verbose=0)
		temp_input.extend(yhat[0].tolist())
		temp_input = temp_input[1:]
		# print(temp_input)
		lst_output.extend(yhat.tolist())
		i = i + 1
	else:
		x_input = x_input.reshape((1, n_steps, 1))
		yhat = model.predict(x_input, verbose=0)
		temp_input.extend(yhat[0].tolist())
		lst_output.extend(yhat.tolist())
		i = i + 1

day_new=np.arange(1,101)
day_pred=np.arange(101,131)


fig3 = plt.figure(figsize = (12,6))
plt.plot(day_new,scaler.inverse_transform(final_df[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.subheader('Next 30 days predictions')
st.pyplot(fig3)

fig4 = plt.figure(figsize = (12,6))
final_Graph=final_df.tolist()
final_Graph.extend(lst_output)
plt.plot(final_Graph[1200:])
st.subheader('Next 30 days predictions with continuation')
st.pyplot(fig4)

fig5 = plt.figure(figsize = (12,6))
final_Graph=scaler.inverse_transform(final_Graph).tolist()
plt.plot(final_Graph)
st.subheader('Complete output graph:')
st.pyplot(fig5)







