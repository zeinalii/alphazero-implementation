#!/usr/bin/env python
import os
import pickle
import collections
import numpy as np
import math
import copy
import torch
import torch.multiprocessing as mp
import datetime
from ..board import encoder_decoder as ed
from ..board.board import Board as ChessBoard
from ..neural_net.neural_net import ChessNet

class UCTNode:
    def __init__(self, game_state, action, parent_node=None):
        self.game_state = game_state  # Current game state
        self.action = action  # Action taken to reach this node
        self.is_expanded = False
        self.parent = parent_node
        self.children = {}
        self.prior_probs = np.zeros(4672, dtype=np.float32)
        self.total_values = np.zeros(4672, dtype=np.float32)
        self.visit_counts = np.zeros(4672, dtype=np.float32)
        self.valid_actions = []
        
    @property
    def visit_count(self):
        return self.parent.visit_counts[self.action]
    
    @visit_count.setter
    def visit_count(self, value):
        self.parent.visit_counts[self.action] = value
    
    @property
    def total_value(self):
        return self.parent.total_values[self.action]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.total_values[self.action] = value
    
    def q_values(self):
        return self.total_values / (1 + self.visit_counts)
    
    def u_values(self):
        return math.sqrt(self.visit_count) * (
            np.abs(self.prior_probs) / (1 + self.visit_counts))
    
    def select_optimal_child(self):
        if self.valid_actions:
            scores = self.q_values() + self.u_values()
            best_action = self.valid_actions[np.argmax(scores[self.valid_actions])]
        else:
            best_action = np.argmax(self.q_values() + self.u_values())
        return best_action
    
    def traverse_to_leaf(self):
        current_node = self
        while current_node.is_expanded:
            optimal_action = current_node.select_optimal_child()
            current_node = current_node.add_child_if_missing(optimal_action)
        return current_node
    
    def inject_dirichlet_noise(self, action_indices, priors):
        valid_priors = priors[action_indices]
        noise = np.random.dirichlet(np.full(len(valid_priors), 0.3))
        priors[action_indices] = 0.75 * valid_priors + 0.25 * noise
        return priors
    
    def expand(self, priors):
        self.is_expanded = True
        valid_actions = []
        adjusted_priors = priors.copy()
        
        for move in self.game_state.actions():
            if move:
                start, end, promotion = move
                action_idx = ed.encode_action(self.game_state, start, end, promotion)
                valid_actions.append(action_idx)
        
        if not valid_actions:
            self.is_expanded = False
            return
        
        self.valid_actions = valid_actions
        
        for idx in range(len(adjusted_priors)):
            if idx not in valid_actions:
                adjusted_priors[idx] = 0.0
        
        if self.parent.parent is None:
            adjusted_priors = self.inject_dirichlet_noise(valid_actions, adjusted_priors)
        
        self.prior_probs = adjusted_priors
    
    def apply_move_and_decode(self, board, move):
        initial, final, promo = ed.decode_action(board, move)
        for i, f, p in zip(initial, final, promo):
            board.player = self.game_state.player
            board.move_piece(i, f, p)
            x_start, y_start = i
            x_end, y_end = f
            piece = board.current_board[x_end, y_end]
            if piece in ["K", "k"] and abs(y_end - y_start) == 2:
                if x_start == 7 and y_end - y_start > 0:
                    board.player = self.game_state.player
                    board.move_piece((7, 7), (7, 5), None)
                elif x_start == 7 and y_end - y_start < 0:
                    board.player = self.game_state.player
                    board.move_piece((7, 0), (7, 3), None)
                elif x_start == 0 and y_end - y_start > 0:
                    board.player = self.game_state.player
                    board.move_piece((0, 7), (0, 5), None)
                elif x_start == 0 and y_end - y_start < 0:
                    board.player = self.game_state.player
                    board.move_piece((0, 0), (0, 3), None)
        return board
                
    def add_child_if_missing(self, move):
        if move not in self.children:
            board_copy = copy.deepcopy(self.game_state)
            board_copy = self.apply_move_and_decode(board_copy, move)
            self.children[move] = UCTNode(board_copy, move, parent_node=self)
        return self.children[move]
    
    def propagate_value(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.visit_count += 1
            if current.game_state.player == 1:
                current.total_value += value_estimate
            elif current.game_state.player == 0:
                current.total_value -= value_estimate
            current = current.parent

class DummyNode:
    def __init__(self):
        self.parent = None
        self.total_values = collections.defaultdict(float)
        self.visit_counts = collections.defaultdict(float)

def UCT_search(game_state, iterations, network):
    root = UCTNode(game_state, move=None, parent_node=DummyNode())
    for _ in range(iterations):
        leaf = root.traverse_to_leaf()
        encoded_state = ed.encode_board(leaf.game_state).transpose(2, 0, 1)
        encoded_tensor = torch.from_numpy(encoded_state).float().cuda()
        priors, value = network(encoded_tensor)
        priors = priors.detach().cpu().numpy().flatten()
        value = value.item()
        
        if leaf.game_state.check_status() and not leaf.game_state.in_check_possible_moves():
            leaf.propagate_value(value)
            continue
        
        leaf.expand(priors)
        leaf.propagate_value(value)
    return np.argmax(root.visit_counts), root

def decode_and_apply_move(board, move):
    initial, final, promo = ed.decode_action(board, move)
    for i, f, p in zip(initial, final, promo):
        board.move_piece(i, f, p)
        x_start, y_start = i
        x_end, y_end = f
        piece = board.current_board[x_end, y_end]
        if piece in ["K", "k"] and abs(y_end - y_start) == 2:
            if x_start == 7 and y_end - y_start > 0:
                board.player = 0
                board.move_piece((7, 7), (7, 5), None)
            elif x_start == 7 and y_end - y_start < 0:
                board.player = 0
                board.move_piece((7, 0), (7, 3), None)
            elif x_start == 0 and y_end - y_start > 0:
                board.player = 1
                board.move_piece((0, 7), (0, 5), None)
            elif x_start == 0 and y_end - y_start < 0:
                board.player = 1
                board.move_piece((0, 0), (0, 3), None)
    return board

def extract_policy(root_node):
    policy_distribution = np.zeros(4672, dtype=np.float32)
    visited_actions = np.where(root_node.visit_counts != 0)[0]
    if root_node.visit_counts.sum() > 0:
        policy_distribution[visited_actions] = root_node.visit_counts[visited_actions] / root_node.visit_counts.sum()
    return policy_distribution

def save_to_pickle(file_name, data):
    file_path = os.path.join(".", "datasets", "iter2", file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(file_name):
    file_path = os.path.join(".", "datasets", file_name)
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def MCTS_self_play(chess_network, total_games, cpu_id):
    for game_id in range(total_games):
        board = ChessBoard()
        is_game_over = False
        training_data = []
        previous_states = []
        game_result = 0
        
        while not is_game_over and board.move_count <= 100:
            repetition_counter = previous_states.count(board.current_board)
            if repetition_counter >= 3:
                break
            previous_states.append(copy.deepcopy(board.current_board))
            
            encoded_board = copy.deepcopy(ed.encode_board(board))
            best_move, root = UCT_search(board, 777, chess_network)
            board = decode_and_apply_move(board, best_move)
            policy = extract_policy(root)
            training_data.append([encoded_board, policy])
            
            print(board.current_board, board.move_count)
            print(" ")
            
            if board.check_status() and not board.in_check_possible_moves():
                game_result = 1 if board.player == 1 else -1
                is_game_over = True
        
        processed_data = []
        for idx, (state, policy) in enumerate(training_data):
            value = 0 if idx == 0 else game_result
            processed_data.append([state, policy, value])
        
        save_filename = f"dataset_cpu{cpu_id}_{game_id}_{datetime.datetime.today().strftime('%Y-%m-%d')}"
        save_to_pickle(save_filename, processed_data)

if __name__ == "__main__":
    network_filename = "current_net_trained8_iter1.pth.tar"
    mp.set_start_method("spawn", force=True)
    chess_net = ChessNet()
    
    if torch.cuda.is_available():
        chess_net.cuda()
    
    chess_net.share_memory()
    chess_net.eval()
    
    print("Initialization complete.")
    
    model_path = os.path.join(".", "model_data", network_filename)
    checkpoint = torch.load(model_path)
    chess_net.load_state_dict(checkpoint['state_dict'])
    
    process_list = []
    for cpu_id in range(6):
        process = mp.Process(target=MCTS_self_play, args=(chess_net, 50, cpu_id))
        process.start()
        process_list.append(process)
    
    for proc in process_list:
        proc.join()
