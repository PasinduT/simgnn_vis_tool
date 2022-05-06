import numpy as np
import torch

def calculateEdgeInfluence(data, model):
    pred = model(data).item()
    edges1 = data['edge_index_1'].detach().numpy()
    print(edges1.shape)
    some1 = []
    for i in range(edges1.shape[1]):
        edges1_copy = torch.from_numpy(np.array([edges1[:, j] for j in range(edges1.shape[1]) if i != j], dtype=np.int64).T).type(torch.long)
        data['edge_index_1'] = edges1_copy

        print(edges1_copy)
        print(edges1)

        some1.append(abs(pred - model(data).item()))

    edges2 = data['edge_index_2'].detach().numpy()
    some2 = []
    for i in range(edges2.shape[1]):
        edges2_copy = torch.from_numpy(np.array([edges2[:, j] for j in range(edges2.shape[1]) if i != j], dtype=np.int64).T).type(torch.long)
        data['edge_index_2'] = edges2_copy

        some2.append(abs(pred - model(data).item()))

    return some1, some2