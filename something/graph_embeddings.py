from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import json
import os
import xml.etree.ElementTree as ET
import torch

import sys
sys.path.append('/Users/pasindutennakoon/Desktop/csc696D/project/')
from SimGNN.src.simgnn import SimGNNTrainer


def loadModel():
    class Args:
        def __init__(self, load_path):
            self.load_path = load_path
            self.batch_size=128
            self.bins=16
            self.bottle_neck_neurons=16
            self.dropout=0.5
            self.epochs=50
            self.filters_1=256
            self.filters_2=128
            self.filters_3=64
            self.histogram=True
            self.learning_rate=0.001
            self.save_path='model.pth'
            self.tensor_neurons=16
            self.testing_graphs='/Users/pasindutennakoon/Desktop/csc696D/project/SimGNN/aids/test/'
            self.training_graphs='/Users/pasindutennakoon/Desktop/csc696D/project/SimGNN/aids/train/'
            self.weight_decay=0.0005

    args = Args('/Users/pasindutennakoon/Desktop/csc696D/project/SimGNN/aids.pth')
    trainer = SimGNNTrainer(args)
    trainer.load()

    print(trainer.score())

    return trainer

def calculateGraphEmbeddings(data, model):
    edge_index_1 = data["edge_index_1"]
    features_1 = data["features_1"]

    abstract_features_1 = model.convolutional_pass(edge_index_1, features_1)

    pooled_features_1 = model.attention(abstract_features_1)

    return pooled_features_1.detach().numpy().flatten().copy()


def calculateProjection(graphEmbeddings):
    # tsne = TSNE(n_components=2)
    # out1 = tsne.fit_transform(node_embeddings)

    pca = PCA(2)
    out1 = pca.fit_transform(graphEmbeddings)

    x = list(map(float, out1[:, 0]))
    y = list(map(float, out1[:, 1]))

    return x, y

def fix_node_id(x):
    if x.startswith('_'):
        return int(x[1:]) - 1
    else:
        raise Exception('something')

def parseGraph(filename):
    tree = ET.parse(filename)

    graph_id = tree.find('.//graph').get('id')


    nodes = {}

    for node in tree.findall('.//node'):
        id = node.get('id')
        properties = {}
        for n in node.findall('./attr'):
            x = list(n)[0]
            if x.tag == 'int':
                val = int(x.text)
            elif x.tag == 'float':
                val = float(x.text)
            elif x.tag == 'string':
                val = x.text.strip()
            properties[n.get('name')] = val

        nodes[fix_node_id(id)] = properties


    edges = []
    for edge in tree.findall('.//edge'):
        f, t = edge.get('from'), edge.get('to')
        
        properties = {}
        for n in edge.findall('./attr'):
            x = list(n)[0]
            if x.tag == 'int':
                val = int(x.text)
            elif x.tag == 'float':
                val = float(x.text)
            elif x.tag == 'string':
                val = x.text.strip
            properties[n.get('name')] = val

        edges.append([fix_node_id(f), fix_node_id(t)])

    return graph_id, nodes, edges

def main():
    dir1 = '../AIDS/'
    dataset = []
    embeddings = []

    trainer = loadModel()
    for filename in os.listdir(dir1):
        if filename.endswith('gxl'):
            graph_id, nodes, edges = parseGraph(dir1 + filename)

            data = {}
            data['graph_1'] = edges
            data['graph_2'] = edges
            data['labels_1'] = [nodes[j]['symbol'] for j in range(len(nodes))]
            data['labels_2'] = [nodes[j]['symbol'] for j in range(len(nodes))]
            data['ged'] = 0

            data = trainer.transfer_to_torch(data)

            embedding = calculateGraphEmbeddings(data, trainer.model)
            embeddings.append(embedding)

            dataset.append({
                'id': graph_id,
                'edges': edges,
                'nodes': [nodes[j]['symbol'] for j in range(len(nodes))]
            })

    with open('dataset.json', 'w') as f:
        json.dump(dataset, f)

    with open('graph_embeddings.json', 'w') as f:
        x, y = calculateProjection(np.array(embeddings))
        json.dump({
            'x': x,
            'y': y
        }, f)

    print(len(dataset), len(embeddings))

if __name__ == '__main__':
    main()