from flask import Flask, g, redirect, render_template, request, session, url_for
import os
import json

import sys



sys.path.append('/Users/pasindutennakoon/Desktop/csc696D/project/')

from SimGNN.src.simgnn import SimGNNTrainer
from .node_embeddings import calculateNodeEmbeddings
from .node_influence import calculateNodeAttentions
from .edge_influence import calculateEdgeInfluence

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

def loadDataset():
    with open('something/dataset.json') as f:
        dataset = json.load(f)

    with open('something/graph_embeddings.json') as f:
        embeddings = json.load(f)

    return dataset, embeddings

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    trainer = loadModel()
    dataset, embeddings = loadDataset()

    @app.route('/')
    def hello():
        return render_template('index.html')


    @app.route('/global', methods=['POST'])
    def globalFn():
        other = {
            'dataset': dataset,
            'embeddings': embeddings
        }
        return other



    @app.route('/something', methods=['POST'])
    def something():
        params = request.json
        print('params', params)
        graph1 = params['graph1']
        graph2 = params['graph2']
        
        data = {
            'graph_1': dataset[graph1]['edges'],
            'graph_2': dataset[graph2]['edges'],
            'labels_1': dataset[graph1]['nodes'],
            'labels_2': dataset[graph2]['nodes'],
            'ged': 0
        }

        data1 = trainer.transfer_to_torch(data)
        prediction = trainer.model(data1).item()

        x, y = calculateNodeEmbeddings(data1, trainer.model)
        node_weight1, node_weight2 = calculateNodeAttentions(data1, trainer.model)
        edge_weight1, edge_weight2 = calculateEdgeInfluence(data1, trainer.model)
        print(x, y)
        # x, y = 0, 0

        other = {
            'data': data,
            'prediction': prediction,
            'x': x,
            'y': y,
            'node_weight1': node_weight1,
            'node_weight2': node_weight2,
            'edge_weight1': edge_weight1,
            'edge_weight2': edge_weight2,
        }
        return other

    return app