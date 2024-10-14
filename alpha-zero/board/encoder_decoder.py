#!/usr/bin/env python

import numpy as np
from .board import Board as ChessBoard

def encode_board(current_board):
    state = current_board.current_board
    encoding = np.zeros((8, 8, 22), dtype=int)
    piece_mapping = {
        "R": 0, "N": 1, "B": 2, "Q": 3, "K": 4, "P": 5,
        "r": 6, "n": 7, "b": 8, "q": 9, "k": 10, "p": 11
    }
    
    for row in range(8):
        for col in range(8):
            piece = state[row, col]
            if piece != " ":
                encoding[row, col, piece_mapping[piece]] = 1
    
    player_layer = 12 if current_board.player == 1 else None
    if player_layer is not None:
        encoding[:, :, player_layer] = 1  # Player's turn

    # Castling rights for White
    if current_board.K_move_count != 0:
        encoding[:, :, 13] = 1  # Queenside castling blocked
        encoding[:, :, 14] = 1  # Kingside castling blocked
    else:
        if current_board.R1_move_count != 0:
            encoding[:, :, 13] = 1
        if current_board.R2_move_count != 0:
            encoding[:, :, 14] = 1

    # Castling rights for Black
    if current_board.k_move_count != 0:
        encoding[:, :, 15] = 1  # Queenside castling blocked
        encoding[:, :, 16] = 1  # Kingside castling blocked
    else:
        if current_board.r1_move_count != 0:
            encoding[:, :, 15] = 1
        if current_board.r2_move_count != 0:
            encoding[:, :, 16] = 1

    # Additional board state features
    encoding[:, :, 17] = current_board.move_count
    encoding[:, :, 18] = current_board.repetitions_w
    encoding[:, :, 19] = current_board.repetitions_b
    encoding[:, :, 20] = current_board.no_progress_count
    encoding[:, :, 21] = current_board.en_passant

    return encoding

def decode_board(encoded_state):
    board_array = np.full((8, 8), " ", dtype=str)
    piece_lookup = {
        0: "R", 1: "N", 2: "B", 3: "Q", 4: "K", 5: "P",
        6: "r", 7: "n", 8: "b", 9: "q", 10: "k", 11: "p"
    }
    
    for row in range(8):
        for col in range(8):
            for piece_idx in range(12):
                if encoded_state[row, col, piece_idx] == 1:
                    board_array[row, col] = piece_lookup[piece_idx]
                    break

    new_board = ChessBoard()
    new_board.current_board = board_array

    # Decode player turn
    if encoded_state[0, 0, 12] == 1:
        new_board.player = 1

    # Decode castling rights for White
    if encoded_state[0, 0, 13] == 1:
        new_board.R1_move_count = 1
    if encoded_state[0, 0, 14] == 1:
        new_board.R2_move_count = 1

    # Decode castling rights for Black
    if encoded_state[0, 0, 15] == 1:
        new_board.r1_move_count = 1
    if encoded_state[0, 0, 16] == 1:
        new_board.r2_move_count = 1

    # Decode additional board state features
    new_board.move_count = encoded_state[0, 0, 17]
    new_board.repetitions_w = encoded_state[0, 0, 18]
    new_board.repetitions_b = encoded_state[0, 0, 19]
    new_board.no_progress_count = encoded_state[0, 0, 20]
    new_board.en_passant = encoded_state[0, 0, 21]

    return new_board

def encode_action(current_board, start_pos, end_pos, promotion=None):
    action_encoding = np.zeros((8, 8, 73), dtype=int)
    i, j = start_pos
    x, y = end_pos
    delta_x, delta_y = x - i, y - j
    piece = current_board.current_board[i, j]
    action_idx = None

    # Queen-like moves
    if piece in {"R", "B", "Q", "K", "P", "r", "b", "q", "k", "p"} and promotion in {None, "queen"}:
        if delta_x != 0 and delta_y == 0:  # Vertical moves
            action_idx = 7 + delta_x if delta_x < 0 else 6 + delta_x
        elif delta_x == 0 and delta_y != 0:  # Horizontal moves
            action_idx = 21 + delta_y if delta_y < 0 else 20 + delta_y
        elif delta_x == delta_y:  # Diagonal NW-SE
            action_idx = 35 + delta_x if delta_x < 0 else 34 + delta_x
        elif delta_x == -delta_y:  # Diagonal NE-SW
            action_idx = 49 + delta_x if delta_x < 0 else 48 + delta_x

    # Knight moves
    elif piece in {"n", "N"}:
        knight_moves = {
            (i + 2, j - 1): 56,
            (i + 2, j + 1): 57,
            (i + 1, j - 2): 58,
            (i - 1, j - 2): 59,
            (i - 2, j + 1): 60,
            (i - 2, j - 1): 61,
            (i - 1, j + 2): 62,
            (i + 1, j + 2): 63
        }
        action_idx = knight_moves.get((x, y))

    # Pawn promotions
    elif piece in {"p", "P"} and (x == 0 or x == 7) and promotion:
        promotion_map = {
            "rook": 64, "knight": 65, "bishop": 66,
            "rook_capture": 67, "knight_capture": 68, "bishop_capture": 69,
            "rook_diag": 70, "knight_diag": 71, "bishop_diag": 72
        }
        if delta_y == 0:
            action_idx = promotion_map.get(f"{promotion}")
        elif delta_y == -1:
            action_idx = promotion_map.get(f"{promotion}_capture")
        elif delta_y == 1:
            action_idx = promotion_map.get(f"{promotion}_diag")

    if action_idx is not None:
        action_encoding[i, j, action_idx] = 1

    flat_index = np.argmax(action_encoding)
    return flat_index

def decode_action(current_board, encoded_action):
    action_vector = np.zeros(4672, dtype=int)
    action_vector[encoded_action] = 1
    reshaped_action = action_vector.reshape(8, 8, 73)
    positions = np.argwhere(reshaped_action == 1)
    
    if not positions.size:
        return [], [], []

    i, j, k = positions[0]
    initial_pos = (i, j)
    promoted_piece = None
    final_pos = None

    # Determine move based on action index
    if 0 <= k <= 13:
        dy = 0
        dx = k - 7 if k < 7 else k - 6
        final_pos = (i + dx, j + dy)
    elif 14 <= k <= 27:
        dx = 0
        dy = k - 21 if k < 21 else k - 20
        final_pos = (i + dx, j + dy)
    elif 28 <= k <= 41:
        dy = k - 35 if k < 35 else k - 34
        dx = dy
        final_pos = (i + dx, j + dy)
    elif 42 <= k <= 55:
        dx = k - 49 if k < 49 else k - 48
        dy = -dx
        final_pos = (i + dx, j + dy)
    elif 56 <= k <= 63:
        knight_moves = {
            56: (i + 2, j - 1),
            57: (i + 2, j + 1),
            58: (i + 1, j - 2),
            59: (i - 1, j - 2),
            60: (i - 2, j + 1),
            61: (i - 2, j - 1),
            62: (i - 1, j + 2),
            63: (i + 1, j + 2)
        }
        final_pos = knight_moves.get(k)
    else:
        promotion_map = {
            64: ("R", (i - 1, j)),
            65: ("N", (i - 1, j)),
            66: ("B", (i - 1, j)),
            67: ("R", (i - 1, j - 1)),
            68: ("N", (i - 1, j - 1)),
            69: ("B", (i - 1, j - 1)),
            70: ("R", (i - 1, j + 1)),
            71: ("N", (i - 1, j + 1)),
            72: ("B", (i - 1, j + 1)),
        }
        if current_board.player == 1:
            promotion_map = {key: (val[0].lower(), (val[1][0] + 2, val[1][1])) for key, val in promotion_map.items()}
        piece, final_pos = promotion_map.get(k, (None, None))
        promoted_piece = piece

    # Auto-promotion to Queen if no promotion piece is specified
    if final_pos and current_board.current_board[i, j] in {"P", "p"} and final_pos[0] in {0, 7} and not promoted_piece:
        promoted_piece = "Q" if current_board.player == 0 else "q"

    return [initial_pos], [final_pos], [promoted_piece]
