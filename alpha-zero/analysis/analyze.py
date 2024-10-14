import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ..board import encoder_decoder as ed
from ..board.visualize import view_board as vb

relative_data_dir = "./datasets/iter2/"
dataset_filename = "dataset_cpu1_5"
full_dataset_path = os.path.join(relative_data_dir, dataset_filename)

with open(full_dataset_path, 'rb') as dataset_file:
    chess_data = pickle.load(dataset_file, encoding='bytes')

final_move_index = np.argmax(chess_data[-1][1])

final_board = ed.decode_board(chess_data[-1][0])
final_action = ed.decode_action(final_board, final_move_index)

final_board.move_piece(final_action[0][0], final_action[1][0], final_action[2][0])

images_dir = os.path.join(".", "gamesimages", "ex4")

os.makedirs(images_dir, exist_ok=True)


for move_num, move_data in enumerate(chess_data):
    current_board = ed.decode_board(move_data[0])
    figure = vb(current_board.current_board)
    image_path = os.path.join(images_dir, f"{dataset_filename}_{move_num}.png")
    plt.savefig(image_path)
    plt.close()  


final_figure = vb(final_board.current_board)
final_image_path = os.path.join(images_dir, f"{dataset_filename}_{move_num + 1}.png")
plt.savefig(final_image_path)
plt.close()
