import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import som_clustering
import modelling
import forecast
from datetime import datetime, date, timedelta
import os


def check_tick(name):
    loc = "./daily/"
    dir = os.listdir(loc)
    files = []
    for d in dir:
        x = d.split(".")
        files.append(x[0])

    my_set = set(files)  # Convert list to set
    return name in my_set


# Web App Title
st.markdown(
    """
# **Stock Forecasting App**

"""
)
st.info(
    f"This is the Stock Forecasting Application created in Streamlit using the SOM library."
)


st.sidebar.title("Stock App")

name = st.sidebar.text_input("Enter Stock Ticker", "AALI")
if check_tick(name) == True:
    st.sidebar.caption(":green[Stock Tick is Valid]")
else:
    st.sidebar.caption(":red[Enter a valid Stock Tick!]")
option = st.sidebar.selectbox(
    "Menu",
    ("Stock Data EDA", "Cluster with SOM", "Build & Evaluate Model", "Forecast"),
)
st.sidebar.warning(
    """
                App feature:
                1. Stock EDA
                2. Cluster Stock with SOM
                3. Build & Evaluate model
                4. Forecasting"""
)
st.sidebar.caption("Developed by: Author")
st.subheader(f"{option} for {name} Stock")

if option == "Stock Data EDA":
    if check_tick(name) == False:
        st.error("Enter a valid Stock Tick!")
    else:
        dataset = pd.read_csv("./daily/" + name + ".csv")
        st.text(f"\n\n{name} Stock Data\n")
        dataset.set_index("timestamp")
        st.dataframe(dataset, use_container_width=True)

        st.text(f"\n\n{name} Closing Price Chart\n")
        st.line_chart(dataset["close"])

if option == "Cluster with SOM":
    # name = st.text_input("Enter Stock Ticker", "AALI")

    # get reference vector and cluster
    if check_tick(name) == False:
        st.error("Enter a valid Stock Tick!")
    else:
        st.write(f"Clustering and calculating reference vector using **SOM**")

        c1, c2, c3, c4 = som_clustering.get_cluster(name)

        st.success(
            f"Finished Clustering and calculating reference vector for **:orange[{name}]** Stock using SOM"
        )
        som_clustering.plot(name)
        st.pyplot(plt.gcf())

if option == "Build & Evaluate Model":
    st.write(f"Building models for **:orange[{name}]** Stock")
    plt.cla()
    c1, c2, c3, c4 = som_clustering.get_cluster(name)
    for i in range(4):
        if i == 0:
            rmse_c1, x, y = modelling.build_model(c1, i, name)
            # with st.expander(f"Display Predicted Price Cluster {i+1}"):
            st.success(
                f"Finished bulding model {i+1}, RMSE for Cluster {i+1}: {rmse_c1}"
            )
            st.pyplot(plt.gcf())

        if i == 1:
            plt.cla()
            rmse_c2, x, y = modelling.build_model(c2, i, name)
            st.success(
                f"Finished bulding model {i+1}, RMSE for Cluster {i+1}: {rmse_c2}"
            )
            st.pyplot(plt.gcf())
        if i == 2:
            plt.cla()
            rmse_c3, x, y = modelling.build_model(c3, i, name)
            st.success(
                f"Finished bulding model {i+1}, RMSE for Cluster {i+1}: {rmse_c3}"
            )
            st.pyplot(plt.gcf())
        if i == 3:
            plt.cla()
            rmse_c4, x, y = modelling.build_model(c4, i, name)
            st.success(
                f"Finished bulding model {i+1}, RMSE for Cluster {i+1}: {rmse_c4}"
            )
            st.pyplot(plt.gcf())

if option == "Forecast":
    if check_tick(name) == False:
        st.error("Enter a valid Stock Tick!")
    else:
        dataset = pd.read_csv("./daily/" + name + ".csv")
        dataset = dataset[["open", "close", "low", "high"]]
        rmse_c1, ap, pp = forecast.load_model(dataset, 4, name)
        ap = np.concatenate((ap, pp), axis=0)
        column_values = ["prices"]

        today = "2023-01-06"
        datetime.strptime(today, "%Y-%m-%d")
        myday = datetime.strptime(today, "%Y-%m-%d")

        dates = []
        for x in range(120):
            abc = myday + timedelta(days=x + 1)
            dates.append(abc)

        pp = pd.DataFrame(data=pp, columns=column_values)
        dd = pd.DataFrame(data=dates, columns=["dates"])

        number = st.number_input(
            "Insert a number of days (max 100 days)", min_value=1, max_value=100
        )

        pp = pp.iloc[int(len(pp) * 0.90) : int(len(pp))]
        pp = pp.reset_index(drop=True)

        pp = pp["prices"][1 : number + 1]

        # st.success(f"Finished bulding model, RMSE: {rmse_c1}")
        # st.pyplot(plt.gcf())
        st.text(f"\n\n{name} Forecast Price\n")
        st.line_chart(data=pp)

        for x in range(number):
            st.success(
                f"Forecast **:orange[{name}]** Stock Price Result for day {x+1} **({dates[x].date()})** is: **:blue[{int(pp[x+1])}]** (USD)"
            )
