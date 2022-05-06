from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


def calculateNodeEmbeddings(data, model):
    edge_index_1 = data["edge_index_1"]
    edge_index_2 = data["edge_index_2"]
    features_1 = data["features_1"]
    features_2 = data["features_2"]

    abstract_features_1 = model.convolutional_pass(edge_index_1, features_1)
    abstract_features_2 = model.convolutional_pass(edge_index_2, features_2)

    node_embeddings1 = abstract_features_1.detach().numpy().copy()
    node_embeddings2 = abstract_features_2.detach().numpy().copy()

    node_embeddings = np.vstack((node_embeddings1, node_embeddings2))

    print(node_embeddings1.shape, node_embeddings2.shape, node_embeddings.shape)

    # tsne = TSNE(n_components=2)
    # out1 = tsne.fit_transform(node_embeddings)

    pca = PCA(2)
    out1 = pca.fit_transform(node_embeddings)

    x = list(map(float, out1[:, 0]))
    y = list(map(float, out1[:, 1]))

    return x, y