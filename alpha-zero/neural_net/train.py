#!/usr/bin/env python


import os
import pickle
import numpy as np
import torch
from .neural_net import ChessNet, train


def train_chessnet(net_to_train="current_net_trained7_iter1.pth.tar", save_as="current_net_trained8_iter1.pth.tar"):
    # Collect datasets from iter1 and iter0 directories
    datasets = []
    for iteration in ["iter1", "iter0"]:
        dataset_dir = os.path.join("datasets", iteration)
        for file_name in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, file_name)
            with open(file_path, 'rb') as file_handle:
                loaded_data = pickle.load(file_handle, encoding='bytes')
                datasets.extend(loaded_data)
    
    datasets = np.array(datasets)
    
    # Initialize and load the neural network
    chess_network = ChessNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chess_network.to(device)
    
    model_path = os.path.join("model_data", net_to_train)
    checkpoint = torch.load(model_path, map_location=device)
    chess_network.load_state_dict(checkpoint['state_dict'])
    
    # Train the network with the collected data
    train(chess_network, datasets)
    
    # Save the trained model
    save_path = os.path.join("model_data", save_as)
    torch.save({'state_dict': chess_network.state_dict()}, save_path)

if __name__ == "__main__":
    train_chessnet()
