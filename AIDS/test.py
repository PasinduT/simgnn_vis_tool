from cgitb import small
import json
import xml.etree.ElementTree as ET
import numpy as np
import random
import pickle

nTest = 300
nTrain = 300

dir1 = '../../astar/graph-matching-toolkit/data/AIDS/'

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

        edges.append((fix_node_id(f), fix_node_id(t), properties))

    return nodes, edges


def main():
    some = np.zeros(50 * 50)
    with open('aids.txt') as f:
        i = 0
        for line in f:
            val = float(line)
            some[i] = val
            i += 1

    some = some.reshape((50, 50))
    print(some)   

    filenames = []
    tree = ET.parse('small.xml')
    for p in tree.findall('.//print'):
        filenames.append(p.get('file'))

    print(filenames, len(filenames))

    
    train_collection = {}

    i = 0
    while i < nTrain:
        first = random.randint(0, 49)
        second = random.randint(0, 49)

        if (first, second) in train_collection or some[first][second] < 0.0:
            continue

        train_collection[(first, second)] = True

        n1, e1 = parseGraph(dir1 + filenames[first])
        n2, e2 = parseGraph(dir1 + filenames[second])

        data = {}
        data['graph_1'] = [(fro, to) for fro, to, prop in e1]
        data['graph_2'] = [(fro, to) for fro, to, prop in e2]
        data['labels_1'] = [n1[j]['symbol'] for j in range(len(n1))]
        data['labels_2'] = [n2[j]['symbol'] for j in range(len(n2))]
        data['ged'] = some[first][second]

        # print(data)

        with open(f'train/{i}.json', 'w') as f:
            json.dump(data, f)

        i += 1


    test_collection = {}

    i = 0
    while i < nTest:
        first = random.randint(0, 49)
        second = random.randint(0, 49)

        if (first, second) in test_collection or some[first][second] < 0.0:
            continue

        test_collection[(first, second)] = True

        n1, e1 = parseGraph(dir1 + filenames[first])
        n2, e2 = parseGraph(dir1 + filenames[second])

        data = {}
        data['graph_1'] = [(fro, to) for fro, to, prop in e1]
        data['graph_2'] = [(fro, to) for fro, to, prop in e2]
        data['labels_1'] = [n1[j]['symbol'] for j in range(len(n1))]
        data['labels_2'] = [n2[j]['symbol'] for j in range(len(n2))]
        data['ged'] = some[first][second]

        # print(data)

        with open(f'test/{i}.json', 'w') as f:
            json.dump(data, f)

        i += 1

    with open('output.log', 'wb') as f:
        pickle.dump((train_collection, test_collection), f)


if __name__ == '__main__':
    main()