import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
import matplotlib.pyplot as plt


def cluster(name):
    dataset = pd.read_csv("./daily/" + name + ".csv")
    # dataset = pd.read_csv(loc + d + "/" + images[i])
    # print(f"{name}.csv")
    # dataset2 = dataset.copy()
    dataset2 = dataset[["open", "close", "low", "high"]]
    dataset2.head()

    data = dataset2.iloc[:, :].values  # Predictor attributes

    som_shape = (2, 2)
    som = MiniSom(
        som_shape[0],
        som_shape[1],
        data.shape[1],
        sigma=0.5,
        learning_rate=0.5,
        random_seed=10,
    )
    som.random_weights_init(data)
    starting_weights = som.get_weights().copy()

    som.train_batch(data, 500)

    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in data]).T
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

    winner_coordinates = np.array([som.winner(x) for x in data]).T

    # for x in data:
    #     print("data ke: ", x, ", clusternya: ", som.winner(x))

    column_title = ["open", "close", "low", "high"]

    df1 = pd.DataFrame([], columns=column_title)
    df2 = pd.DataFrame([], columns=column_title)
    df3 = pd.DataFrame([], columns=column_title)
    df4 = pd.DataFrame([], columns=column_title)
    # cluster1 = []
    # df0 = pd.DataFrame([], columns=column_title)

    for y in range(4):
        if y == 0:
            for x in data[cluster_index == y]:
                df1.loc[len(df1)] = x
        elif y == 1:
            for x in data[cluster_index == y]:
                df2.loc[len(df2)] = x
        elif y == 2:
            for x in data[cluster_index == y]:
                df3.loc[len(df3)] = x
        elif y == 3:
            for x in data[cluster_index == y]:
                df4.loc[len(df4)] = x

    return df1, df2, df3, df4, data, cluster_index, som


def get_cluster(name):
    a, b, c, d, e, f, g = cluster(name)
    return a, b, c, d


def plot(name):
    a, b, c, d, e, f, g = cluster(name)

    # print(df4)
    plt.clf()
    # plotting the clusters using the first 2 dimentions of the data
    for c in np.unique(f):
        plt.scatter(
            e[f == c, 0],
            e[f == c, 1],
            label="cluster=" + str(c + 1),
            alpha=0.7,
        )
    # plotting centroids
    for centroid in g.get_weights():
        plt.scatter(
            centroid[:, 0],
            centroid[:, 1],
            marker="x",
            s=80,
            linewidths=5,
            color="k",
            label="centroid",
        )
    plt.title(f"SOM Cluster {name} Stock Data")
    plt.legend()
    plt.savefig(f"./figure/SOM-Cluster-{name}-Stock-Data.png")
    plt.show()


############################

# dataset = pd.read_csv("AALI.csv")
# # dataset2 = dataset.copy()
# dataset2 = dataset[["open", "close", "low", "high"]]
# dataset2.head()

# data = dataset2.iloc[:, :].values  # Predictor attributes

# som_shape = (2, 2)
# som = MiniSom(
#     som_shape[0],
#     som_shape[1],
#     data.shape[1],
#     sigma=0.5,
#     learning_rate=0.5,
#     random_seed=10,
# )
# som.random_weights_init(data)
# starting_weights = som.get_weights().copy()

# som.train_batch(data, 500)

# # each neuron represents a cluster
# winner_coordinates = np.array([som.winner(x) for x in data]).T
# # with np.ravel_multi_index we convert the bidimensional
# # coordinates to a monodimensional index
# cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

# import matplotlib.pyplot as plt

# # plotting the clusters using the first 2 dimentions of the data
# for c in np.unique(cluster_index):
#     plt.scatter(
#         data[cluster_index == c, 0],
#         data[cluster_index == c, 1],
#         label="cluster=" + str(c + 1),
#         alpha=0.7,
#     )

# # for x in data[cluster_index == 8]:
# #     print(x)

# # for c in np.unique(cluster_index):
# #     print(c)

# # print(winner_coordinates)
# # with np.printoptions(edgeitems=500):
# #     print(winner_coordinates)

# # plotting centroids
# for centroid in som.get_weights():
#     plt.scatter(
#         centroid[:, 0],
#         centroid[:, 1],
#         marker="x",
#         s=80,
#         linewidths=5,
#         color="k",
#         label="centroid",
#     )
# plt.legend()
# plt.show()

# winner_coordinates = np.array([som.winner(x) for x in data]).T

# # for x in data:
# #     print("data ke: ", x, ", clusternya: ", som.winner(x))

# column_title = ["open", "close", "low", "high"]

# # cluster1 = []
# # df0 = pd.DataFrame([], columns=column_title)
# df1 = pd.DataFrame([], columns=column_title)
# df2 = pd.DataFrame([], columns=column_title)
# df3 = pd.DataFrame([], columns=column_title)
# df4 = pd.DataFrame([], columns=column_title)

# for y in range(8):
#     if y == 0:
#         for x in data[cluster_index == y]:
#             df1.loc[len(df1)] = x
#     elif y == 1:
#         for x in data[cluster_index == y]:
#             df2.loc[len(df2)] = x
#     elif y == 2:
#         for x in data[cluster_index == y]:
#             df3.loc[len(df3)] = x
#     elif y == 3:
#         for x in data[cluster_index == y]:
#             df4.loc[len(df4)] = x

# # print(df4)

############################
