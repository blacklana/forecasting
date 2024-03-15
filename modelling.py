import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import datetime as dt

# import som_clustering
from sklearn.metrics import mean_squared_error


warnings.filterwarnings("ignore")

dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d")


def build_model(cluster, i, name):
    training_data = pd.DataFrame(cluster["close"][0 : int(len(cluster) * 0.70)])
    testing_data = pd.DataFrame(
        cluster["close"][int(len(cluster) * 0.70) : int(len(cluster))]
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(training_data["close"].values.reshape(-1, 1))

    prediction_days = 10

    x_train = []
    y_train = []

    for x in range(prediction_days, len(df_scaled)):
        x_train.append(df_scaled[x - prediction_days : x, 0])
        y_train.append(df_scaled[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)

    actual_price = testing_data["close"].values

    total_dataset = pd.concat((training_data["close"], testing_data["close"]), axis=0)

    model_inputs = total_dataset[
        len(total_dataset) - len(testing_data) - prediction_days :
    ].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days : x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)

    predicted_prices = scaler.inverse_transform(predicted_prices)

    # print(f"Train Score Cluster {i}: RMSE {trainScore}")
    plt.plot(actual_price, color="black", label="Actual Prices")
    plt.plot(predicted_prices, color="blue", label="Predicted Prices")
    plt.title(f"Predicte Prices {name} Stock for Cluster {i+1}")
    plt.xlabel("Time")
    plt.ylabel(f"{name} Stock Prices")
    plt.legend()
    plt.savefig(f"./figure/Predicted-Prices-{name}Stock-for-Cluster-{i+1}.png")
    plt.show()

    # predicted_prices = scaler.inverse_transform(predicted_prices)
    actual_price = scaler.inverse_transform([actual_price])

    trainScore = np.sqrt(mean_squared_error(actual_price[0], predicted_prices[:, 0]))

    # real_data = [
    #     model_inputs[len[model_inputs] + 1 - prediction_days : len(model_inputs + 1), 0]
    # ]
    # real_data = np.array(real_data)
    # real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    # prediction = model.predict(real_data)
    # prediction = scaler.inverse_transform(prediction)

    # print(f"Preediction Price: {prediction}")
    model_name = "./models/model_" + str(name) + "_" + str(i + 1) + ".h5"
    # model.save(model_name)
    return trainScore, actual_price, predicted_prices
