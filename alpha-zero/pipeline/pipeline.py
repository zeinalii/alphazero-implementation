#!/usr/bin/env python

import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from ..neural_net.neural_net import ChessNet, train
from ..mcts.mcts import MCTS_self_play

def aggregate_datasets(iter_dirs):
    combined_data = []
    for directory in iter_dirs:
        dataset_dir = os.path.join(".", "datasets", f"iter{directory}")
        for filename in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, filename)
            with open(file_path, 'rb') as file:
                combined_data.extend(pickle.load(file, encoding='bytes'))
    return np.array(combined_data)

if __name__ == "__main__":
    for epoch in range(10):
        # Phase 1: Execute MCTS
        mcts_model = "current_net_trained8_iter1.pth.tar"
        mp.set_start_method("spawn", force=True)
        
        model = ChessNet()
        device_available = torch.cuda.is_available()
        if device_available:
            model.cuda()
        model.share_memory()
        model.eval()
        
        print("Initializing MCTS phase...")
        model_path = os.path.join(".", "model_data", mcts_model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        
        mcts_procs = []
        for proc_id in range(6):
            process = mp.Process(target=MCTS_self_play, args=(model, 50, proc_id))
            process.start()
            mcts_procs.append(process)
        for proc in mcts_procs:
            proc.join()
        
        # Phase 2: Train the Network
        training_model = "current_net_trained8_iter1.pth.tar"
        save_model = "current_net_trained8_iter1.pth.tar"
        
        iterations = [0, 1, 2]
        training_data = aggregate_datasets(iterations)
        
        mp.set_start_method("spawn", force=True)
        trainer = ChessNet()
        if device_available:
            trainer.cuda()
        trainer.share_memory()
        trainer.train()
        
        print("Starting training phase...")
        training_path = os.path.join(".", "model_data", training_model)
        trainer_checkpoint = torch.load(training_path)
        trainer.load_state_dict(trainer_checkpoint['state_dict'])
        
        training_procs = []
        for proc_id in range(6):
            proc = mp.Process(target=train, args=(trainer, training_data, 0, 200, proc_id))
            proc.start()
            training_procs.append(proc)
        for proc in training_procs:
            proc.join()
        
        # Save the updated model
        save_path = os.path.join(".", "model_data", save_model)
        torch.save({'state_dict': trainer.state_dict()}, save_path)
