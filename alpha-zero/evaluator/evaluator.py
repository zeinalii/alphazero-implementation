#!/usr/bin/env python

import os
import copy
import pickle
import torch
import numpy as np
import torch.multiprocessing as mp
from ..neural_net.neural_net import ChessNet as cnet
from ..board.board import Board as c_board
from ..board import encoder_decoder as ed
from ..mcts.mcts import UCT_search, do_decode_n_move_pieces

def save_as_pickle(filename, data):
    directory = os.path.join("evaluator_data", filename)
    with open(directory, 'wb') as file:
        pickle.dump(data, file)

class arena():
    def __init__(self, current_chessnet, best_chessnet):
        self.current_net = current_chessnet
        self.best_net = best_chessnet
    
    def play_round(self):
        if np.random.rand() < 0.5:
            white_player, black_player = self.current_net, self.best_net
            white_label, black_label = "current", "best"
        else:
            white_player, black_player = self.best_net, self.current_net
            white_label, black_label = "best", "current"
        
        game_board = c_board()
        is_finished = False
        history_states = []
        game_data = []
        outcome = 0

        while not is_finished and game_board.move_count <= 100:
            repetition = sum(np.array_equal(game_board.current_board, state) for state in history_states)
            if repetition == 3:
                break  # Draw by threefold repetition
            history_states.append(copy.deepcopy(game_board.current_board))
            encoded_state = copy.deepcopy(ed.encode_board(game_board))
            game_data.append(encoded_state)

            if game_board.player == 0:
                selected_move, _ = UCT_search(game_board, 777, white_player)
            else:
                selected_move, _ = UCT_search(game_board, 777, black_player)
            
            game_board = do_decode_n_move_pieces(game_board, selected_move)
            print(game_board.current_board, game_board.move_count)
            print(" ")
            
            if game_board.check_status() and not game_board.in_check_possible_moves():
                outcome = -1 if game_board.player == 0 else 1
                is_finished = True

        game_data.append(outcome)
        if outcome == -1:
            return black_label, game_data
        elif outcome == 1:
            return white_label, game_data
        else:
            return None, game_data
    
    def evaluate(self, num_games, cpu_id):
        current_wins = 0
        for game_idx in range(num_games):
            winner, data = self.play_round()
            print(f"{winner} wins!")
            data.append(winner)
            if winner == "current":
                current_wins += 1
            save_as_pickle(f"evaluate_net_dataset_cpu{cpu_id}_{game_idx}", data)
        win_ratio = current_wins / num_games
        print(f"Current_net wins ratio: {win_ratio:.3f}")
        # Optionally save the best network if performance is satisfactory
        # if win_ratio > 0.55:
        #     torch.save({'state_dict': self.current_net.state_dict()}, os.path.join("model_data", "best_net.pth.tar"))

def fork_process(arena_instance, num_games, cpu_id):
    arena_instance.evaluate(num_games, cpu_id)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    current_model_filename = "current_net.pth.tar"
    best_model_filename = "current_net_trained.pth.tar"
    current_model_path = os.path.join("model_data", current_model_filename)
    best_model_path = os.path.join("model_data", best_model_filename)
    
    current_network = cnet()
    best_network = cnet()
    
    # Load the current network
    current_checkpoint = torch.load(current_model_path)
    current_network.load_state_dict(current_checkpoint['state_dict'])
    
    # Load the best network
    best_checkpoint = torch.load(best_model_path)
    best_network.load_state_dict(best_checkpoint['state_dict'])
    
    # Move networks to GPU if available
    if torch.cuda.is_available():
        current_network.cuda()
        best_network.cuda()
    
    # Set networks to evaluation mode
    current_network.eval()
    best_network.eval()
    
    # Share networks across processes
    current_network.share_memory()
    best_network.share_memory()
    
    process_list = []
    for cpu_index in range(6):
        arena_instance = arena(current_network, best_network)
        process = mp.Process(target=fork_process, args=(arena_instance, 50, cpu_index))
        process.start()
        process_list.append(process)
    
    for proc in process_list:
        proc.join()
