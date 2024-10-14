#!/usr/bin/env python

import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from .neural_net import ChessNet, train

def load_dataset(directory):
    data_items = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'rb') as file_handle:
            data_items.extend(pickle.load(file_handle, encoding='bytes'))
    return np.array(data_items)

def initialize_network(model_path):
    model = ChessNet()
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    model.train()
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def main():
    model_to_load = "current_net_trained2.pth.tar"
    model_save_name = "current_net_trained_iter1.pth.tar"
    
    # Data collection
    dataset_directory = os.path.join(".", "datasets", "iter1")
    training_data = load_dataset(dataset_directory)
    
    # Network setup
    mp.set_start_method("spawn", force=True)
    chess_model = initialize_network(os.path.join(".", "model_data", model_to_load))
    
    # Launch training processes
    process_list = []
    for process_id in range(6):
        proc = mp.Process(target=train, args=(chess_model, training_data, 0, 200, process_id))
        proc.start()
        process_list.append(proc)
    
    for proc in process_list:
        proc.join()
    
    # Save the trained model
    torch.save({'state_dict': chess_model.state_dict()},
               os.path.join(".", "model_data", model_save_name))

if __name__ == "__main__":
    main()
