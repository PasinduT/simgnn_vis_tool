import torch
import numpy as np

def calculateNodeAttentions(data, model): 
    edge_index_1 = data["edge_index_1"]
    edge_index_2 = data["edge_index_2"]
    features_1 = data["features_1"]
    features_2 = data["features_2"]

    abstract_features_1 = model.convolutional_pass(edge_index_1, features_1)
    abstract_features_2 = model.convolutional_pass(edge_index_2, features_2)

    attention = model.attention
    embedding1 = abstract_features_1
    att_global_context1 = torch.mean(torch.matmul(embedding1, attention.weight_matrix), dim=0)
    att_transformed_global1 = torch.tanh(att_global_context1)
    att_sigmoid_scores1 = torch.sigmoid(torch.mm(embedding1, att_transformed_global1.view(-1, 1)))



    attention = model.attention
    embedding2 = abstract_features_2
    att_global_context2 = torch.mean(torch.matmul(embedding2, attention.weight_matrix), dim=0)
    att_transformed_global2 = torch.tanh(att_global_context2)
    att_sigmoid_scores2 = torch.sigmoid(torch.mm(embedding2, att_transformed_global2.view(-1, 1)))



    attention_values = att_sigmoid_scores1.detach().numpy().copy()
    maxi = np.max(attention_values)
    mini = np.min(attention_values)
    color_map1 = ((attention_values - mini)/(maxi - mini)).flatten()


    attention_values = att_sigmoid_scores2.detach().numpy().copy()
    maxi = np.max(attention_values)
    mini = np.min(attention_values)
    color_map2 = ((attention_values - mini)/(maxi - mini)).flatten()
    
    return list(map(float, color_map1)), list(map(float, color_map2))