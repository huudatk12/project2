import math
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from PIL import Image
from tensorflow.keras.models import load_model

def color_nagative(val):
	if 0 <= val <15.4:
		color = 'green'
	elif 15.5 <= val <35.4:
		color = 'yellow'
	elif 35.5 <= val <55.4:
		color = 'orange'
	elif 55.5 <= val <140.4:
		color = 'red'
	elif 140.5 <= val < 210.4:
		color = 'violet'
	else:
		color = 'Plum'
	return 'background-color: %s' % color


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()
prediction =st.beta_container()
interative = st.beta_container()
image = Image.open('F:/MachineLearning/Multi_step_timeseries/image/hanoi.jpg')
image2 = Image.open('F:/MachineLearning/Multi_step_timeseries/image/tphcm.jpg')

with header:
	st.title("PM2.5 Prediction Web App")
	st.write("""Dự báo nồng độ ô nhiệm PM2.5 tại thành phố Hà Nội và thành phố Hồ Chí Minh bằng LSTM""")

with dataset:
	st.header('Data input:')
	option = st.sidebar.selectbox('Lựa chọn thành phố', (' ','HN', 'TPHCM'))
	if option == 'HN':
		dataset = pd.read_csv('F:/MachineLearning/Multi_step_timeseries/data/HN.csv')
		st.write(dataset)
		st.sidebar.image(image, caption='TP. Hà Nội')
		st.subheader('Biểu đồ biểu diễn 50 giá trị dại diện cho dữ liệu nồng độ PM2.5 tại TP Hà Nội /theo ngày')
		pm25 = pd.DataFrame(dataset['pm25'].value_counts()).head(50)
		st.bar_chart(pm25)
	elif option == 'TPHCM':
		dataset = pd.read_csv('F:/MachineLearning/Multi_step_timeseries/data/TPHCM.csv')
		# header = 0, infer_datetime_format = True, index_col = 0
		st.write(dataset)
		st.sidebar.image(image2, caption='TP. Hồ Chí Minh')

		st.subheader('Biểu đồ biểu diễn 50 giá trị dại diện cho dữ liệu nồng độ PM2.5 tại TP Hồ Chí Minh /theo ngày')
		pm25 = pd.DataFrame(dataset['pm25'].value_counts()).head(50)
		st.bar_chart(pm25)
	else:
		'None'
with features:
	st.header('Các thuộc tính:')
	st.markdown('* **PM2.5:**  Nồng độ bụi mịn có đường kính nhỏ hơn 2.5 μm trong không khí ')

with modelTraining:
	st.sidebar.header('Huấn luyện mô hình LSTM')
	st.sidebar.text('Lựa chọn các siêu tham số cho mô hình')
	n_part = st.sidebar.slider('Số dữ liệu đầu vào?', min_value=0,max_value=70,value=0,step=7)
	if option == 'HN' or option == 'TPHCM':
		if n_part != 0:
			# data_load_state = st.sidebar.text("Loading....")
			cols = list(dataset)[1:2]
			dataset_for_train_test = dataset[cols].astype(float)

			n_train_day = 365 * 4
			dataset_for_training = dataset_for_train_test.iloc[:n_train_day, :]
			dataset_for_testing = dataset_for_train_test.iloc[n_train_day:, :]

			scaled = StandardScaler()
			scaled = scaled.fit(dataset_for_training)
			dataset_for_training_scaled = scaled.transform(dataset_for_training)
			scaled = scaled.fit(dataset_for_testing)
			dataset_for_testing_scaled = scaled.transform(dataset_for_testing)

			trainX = []
			trainY = []
			n_future = 1

			for i in range(n_part, len(dataset_for_training_scaled) - n_future + 1):
				trainX.append(dataset_for_training_scaled[i - n_part:i, 0:dataset_for_training.shape[1]])
				trainY.append(dataset_for_training_scaled[i + n_future - 1:i + n_future, 0])

			trainX, trainY = np.array(trainX), np.array(trainY)
			model = Sequential()
			model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
			model.add(LSTM(32, activation='relu', return_sequences=False))
			model.add(Dropout(0.2))
			model.add(Dense(trainY.shape[1]))
			model.compile(optimizer='adam', loss='mae')
			model.summary()
			history = model.fit(trainX, trainY, epochs=30, batch_size=16, validation_split=0.1, verbose=0)
			model.save('F:/MachineLearning/Multi_step_timeseries/model/lstm_model.h5')
			yhat = model
			# data_load_state.text('Loading....done!')
		else:
			'None'
	else:
		st.sidebar.text('>>>Tên thành phố đang bị bỏ trống !')

with prediction:
	st.sidebar.header('Dự đoán với mô hình LSTM')
	st.sidebar.text('Lựa chọn các siêu tham số cho dự đoán')
	n_future = st.sidebar.slider('Số ngày muốn dự đoán', min_value=0, max_value=100, value=0, step=1)
	if option == 'HN' or option == 'TPHCM':
		if n_part != 0:
			if n_future != 0:
				model1 = load_model('F:/MachineLearning/Multi_step_timeseries/model/lstm_model.h5')
				train_dates = pd.to_datetime(dataset['date'])

				forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()
				forecast = model1.predict(trainX[-n_future:])

				trainY_inv = scaled.inverse_transform([trainY[-n_future:]])
				forecast_inv = scaled.inverse_transform([forecast])

				nsamples, nx, ny = forecast_inv.shape
				forecast_inv2 = forecast_inv.reshape((nsamples, nx * ny))

				nsamples, nx, ny = trainY_inv[-n_future:].shape
				trainY_inv2 = trainY_inv.reshape((nsamples, nx * ny))

				st.header('Đánh giá mô hình')
				a = math.sqrt(mean_squared_error(trainY_inv2[-n_future:], forecast_inv2))
				b = mean_absolute_error(trainY_inv2[-n_future:], forecast_inv2)
				st.subheader('Sai số bình phương trung bình của mô hình:')
				st.write(a)
				st.subheader('Sai số tuyệt đối trung bình:')
				st.write(b)

				forecast_copies = np.repeat(forecast, dataset_for_training.shape[1], axis=-1)
				y_pred_future = scaled.inverse_transform(forecast_copies)[:, 0]

				forecast_dates = []
				for time_i in forecast_period_dates:
					forecast_dates.append(time_i.date())

				df_forecast = pd.DataFrame({'date': np.array(forecast_dates), 'pm25': y_pred_future})
				df_forecast['date'] = pd.to_datetime(df_forecast['date']).dt.date

				original = dataset[['date', 'pm25']]
				original['date'] = pd.to_datetime(original['date'])
				original = original.loc[original['date'] >= '2020-4-1']

			else:
				'None'
		else:
			st.sidebar.text('>>>Mô hình chưa được huấn luyện !')
	else:
		st.sidebar.text('>>>Tên thành phố đang bị bỏ trống !')


with interative:
	st.header('Kết quả dự báo')
	left1,right1 = st.beta_columns(2)
	if n_future != 0:
		left1.header('Data output:')
		df = pd.DataFrame(df_forecast)
		left1.dataframe(df.style.applymap(color_nagative,subset=['pm25']))
		right1.header('Các mức ảnh hưởng của PM2.5:')
		right1.markdown('* **Green:** Tốt. ')
		right1.markdown('* **Yellow:** Vừa phải. ')
		right1.markdown('* **Orange:** Không tốt cho nhóm người nhạy cảm. ')
		right1.markdown('* **Red:** Không tốt cho sức khỏe.')
		right1.markdown('* **Violet:** Rất không tốt cho sưc khỏe. ')
		right1.markdown('* **Plum:** Nguy hiểm. ')

		st.header('Biểu đồ')
		fig = plt.figure()
		sns.lineplot(original['date'], original['pm25'])
		sns.lineplot(df_forecast['date'], df_forecast['pm25'])
		st.pyplot(fig)
	else:
		'None'