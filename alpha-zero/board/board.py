import numpy as np
import itertools
import copy

class board():
    def __init__(self):
        self.init_board = np.full((8, 8), " ", dtype=str)
        # Setup white pieces
        self.init_board[0, 0] = "r"
        self.init_board[0, 1] = "n"
        self.init_board[0, 2] = "b"
        self.init_board[0, 3] = "q"
        self.init_board[0, 4] = "k"
        self.init_board[0, 5] = "b"
        self.init_board[0, 6] = "n"
        self.init_board[0, 7] = "r"
        self.init_board[1, :] = "p"
        # Setup black pieces
        self.init_board[7, 0] = "R"
        self.init_board[7, 1] = "N"
        self.init_board[7, 2] = "B"
        self.init_board[7, 3] = "Q"
        self.init_board[7, 4] = "K"
        self.init_board[7, 5] = "B"
        self.init_board[7, 6] = "N"
        self.init_board[7, 7] = "R"
        self.init_board[6, :] = "P"
        # Initialize empty squares
        self.init_board[self.init_board == "0.0"] = " "
        
        # Game state variables
        self.move_count = 0
        self.no_progress_count = 0
        self.repetitions_w = 0
        self.repetitions_b = 0
        self.move_history = None
        self.en_passant = -999
        self.en_passant_move = 0
        self.r1_move_count = 0  # Black queenside rook
        self.r2_move_count = 0  # Black kingside rook
        self.k_move_count = 0    # Black king
        self.R1_move_count = 0  # White queenside rook
        self.R2_move_count = 0  # White kingside rook
        self.K_move_count = 0    # White king
        self.current_board = self.init_board.copy()
        # Backup variables for move validation
        self.en_passant_move_copy = None
        self.copy_board = None
        self.en_passant_copy = None
        self.r1_move_count_copy = None
        self.r2_move_count_copy = None
        self.k_move_count_copy = None
        self.R1_move_count_copy = None
        self.R2_move_count_copy = None
        self.K_move_count_copy = None
        # Player turn: 0 for white, 1 for black
        self.player = 0

    def move_rules_P(self, position):
        row, col = position
        possible_moves = []
        threats = []
        board_state = self.current_board

        # Potential threat squares
        if 0 <= row - 1 <= 7 and 0 <= col + 1 <= 7:
            threats.append((row - 1, col + 1))
        if 0 <= row - 1 <= 7 and 0 <= col - 1 <= 7:
            threats.append((row - 1, col - 1))
        
        # Initial pawn moves
        if row == 6:
            if board_state[row - 1, col] == " ":
                possible_moves.append((row - 1, col))
                if board_state[row - 2, col] == " ":
                    possible_moves.append((row - 2, col))
        # En passant possibilities
        elif row == 3 and self.en_passant != -999:
            if col - 1 == self.en_passant and abs(self.en_passant_move - self.move_count) == 1:
                possible_moves.append((row - 1, col - 1))
            if col + 1 == self.en_passant and abs(self.en_passant_move - self.move_count) == 1:
                possible_moves.append((row - 1, col + 1))
        # Regular pawn moves
        if 1 <= row <= 5 and board_state[row - 1, col] == " ":
            possible_moves.append((row - 1, col))
        # Captures
        if col == 0 and board_state[row - 1, col + 1] in ["r", "n", "b", "q", "k", "p"]:
            possible_moves.append((row - 1, col + 1))
        elif col == 7 and board_state[row - 1, col - 1] in ["r", "n", "b", "q", "k", "p"]:
            possible_moves.append((row - 1, col - 1))
        elif 1 <= col <= 6:
            if board_state[row - 1, col + 1] in ["r", "n", "b", "q", "k", "p"]:
                possible_moves.append((row - 1, col + 1))
            if board_state[row - 1, col - 1] in ["r", "n", "b", "q", "k", "p"]:
                possible_moves.append((row - 1, col - 1))
        
        return possible_moves, threats

    def move_rules_p(self, position):
        row, col = position
        possible_moves = []
        threats = []
        board_state = self.current_board

        # Potential threat squares
        if 0 <= row + 1 <= 7 and 0 <= col + 1 <= 7:
            threats.append((row + 1, col + 1))
        if 0 <= row + 1 <= 7 and 0 <= col - 1 <= 7:
            threats.append((row + 1, col - 1))
        
        # Initial pawn moves
        if row == 1:
            if board_state[row + 1, col] == " ":
                possible_moves.append((row + 1, col))
                if board_state[row + 2, col] == " ":
                    possible_moves.append((row + 2, col))
        # En passant possibilities
        elif row == 4 and self.en_passant != -999:
            if col - 1 == self.en_passant and abs(self.en_passant_move - self.move_count) == 1:
                possible_moves.append((row + 1, col - 1))
            if col + 1 == self.en_passant and abs(self.en_passant_move - self.move_count) == 1:
                possible_moves.append((row + 1, col + 1))
        # Regular pawn moves
        if 2 <= row <= 6 and board_state[row + 1, col] == " ":
            possible_moves.append((row + 1, col))
        # Captures
        if col == 0 and board_state[row + 1, col + 1] in ["R", "N", "B", "Q", "K", "P"]:
            possible_moves.append((row + 1, col + 1))
        elif col == 7 and board_state[row + 1, col - 1] in ["R", "N", "B", "Q", "K", "P"]:
            possible_moves.append((row + 1, col - 1))
        elif 1 <= col <= 6:
            if board_state[row + 1, col + 1] in ["R", "N", "B", "Q", "K", "P"]:
                possible_moves.append((row + 1, col + 1))
            if board_state[row + 1, col - 1] in ["R", "N", "B", "Q", "K", "P"]:
                possible_moves.append((row + 1, col - 1))
        
        return possible_moves, threats

    def move_rules_r(self, position):
        row, col = position
        board = self.current_board
        moves = []

        # Upwards
        r = row - 1
        while r >= 0:
            if board[r, col] != " ":
                if board[r, col] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, col))
                break
            moves.append((r, col))
            r -= 1

        # Downwards
        r = row + 1
        while r <= 7:
            if board[r, col] != " ":
                if board[r, col] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, col))
                break
            moves.append((r, col))
            r += 1

        # Right
        c = col + 1
        while c <= 7:
            if board[row, c] != " ":
                if board[row, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((row, c))
                break
            moves.append((row, c))
            c += 1

        # Left
        c = col - 1
        while c >= 0:
            if board[row, c] != " ":
                if board[row, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((row, c))
                break
            moves.append((row, c))
            c -= 1

        return moves

    def move_rules_R(self, position):
        row, col = position
        board = self.current_board
        moves = []

        # Upwards
        r = row - 1
        while r >= 0:
            if board[r, col] != " ":
                if board[r, col] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, col))
                break
            moves.append((r, col))
            r -= 1

        # Downwards
        r = row + 1
        while r <= 7:
            if board[r, col] != " ":
                if board[r, col] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, col))
                break
            moves.append((r, col))
            r += 1

        # Right
        c = col + 1
        while c <= 7:
            if board[row, c] != " ":
                if board[row, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((row, c))
                break
            moves.append((row, c))
            c += 1

        # Left
        c = col - 1
        while c >= 0:
            if board[row, c] != " ":
                if board[row, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((row, c))
                break
            moves.append((row, c))
            c -= 1

        return moves

    def move_rules_n(self, position):
        row, col = position
        potential_moves = [
            (row + 2, col - 1), (row + 2, col + 1),
            (row + 1, col - 2), (row - 1, col - 2),
            (row - 2, col + 1), (row - 2, col - 1),
            (row - 1, col + 2), (row + 1, col + 2)
        ]
        valid_moves = []
        board = self.current_board

        for r, c in potential_moves:
            if 0 <= r <= 7 and 0 <= c <= 7:
                if board[r, c] in ["R", "N", "B", "Q", "K", "P", " "]:
                    valid_moves.append((r, c))
        
        return valid_moves

    def move_rules_N(self, position):
        row, col = position
        potential_moves = [
            (row + 2, col - 1), (row + 2, col + 1),
            (row + 1, col - 2), (row - 1, col - 2),
            (row - 2, col + 1), (row - 2, col - 1),
            (row - 1, col + 2), (row + 1, col + 2)
        ]
        valid_moves = []
        board = self.current_board

        for r, c in potential_moves:
            if 0 <= r <= 7 and 0 <= c <= 7:
                if board[r, c] in ["r", "n", "b", "q", "k", "p", " "]:
                    valid_moves.append((r, c))
        
        return valid_moves

    def move_rules_b(self, position):
        row, col = position
        board = self.current_board
        moves = []

        # Top-left diagonal
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0:
            if board[r, c] != " ":
                if board[r, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r -= 1
            c -= 1

        # Bottom-right diagonal
        r, c = row + 1, col + 1
        while r <= 7 and c <= 7:
            if board[r, c] != " ":
                if board[r, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r += 1
            c += 1

        # Top-right diagonal
        r, c = row - 1, col + 1
        while r >= 0 and c <= 7:
            if board[r, c] != " ":
                if board[r, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r -= 1
            c += 1

        # Bottom-left diagonal
        r, c = row + 1, col - 1
        while r <= 7 and c >= 0:
            if board[r, c] != " ":
                if board[r, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r += 1
            c -= 1

        return moves

    def move_rules_B(self, position):
        row, col = position
        board = self.current_board
        moves = []

        # Top-left diagonal
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0:
            if board[r, c] != " ":
                if board[r, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r -= 1
            c -= 1

        # Bottom-right diagonal
        r, c = row + 1, col + 1
        while r <= 7 and c <= 7:
            if board[r, c] != " ":
                if board[r, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r += 1
            c += 1

        # Top-right diagonal
        r, c = row - 1, col + 1
        while r >= 0 and c <= 7:
            if board[r, c] != " ":
                if board[r, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r -= 1
            c += 1

        # Bottom-left diagonal
        r, c = row + 1, col - 1
        while r <= 7 and c >= 0:
            if board[r, c] != " ":
                if board[r, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r += 1
            c -= 1

        return moves

    def move_rules_q(self, position):
        row, col = position
        board = self.current_board
        moves = []

        # Rook-like movements
        # Upwards
        r = row - 1
        while r >= 0:
            if board[r, col] != " ":
                if board[r, col] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, col))
                break
            moves.append((r, col))
            r -= 1

        # Downwards
        r = row + 1
        while r <= 7:
            if board[r, col] != " ":
                if board[r, col] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, col))
                break
            moves.append((r, col))
            r += 1

        # Right
        c = col + 1
        while c <= 7:
            if board[row, c] != " ":
                if board[row, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((row, c))
                break
            moves.append((row, c))
            c += 1

        # Left
        c = col - 1
        while c >= 0:
            if board[row, c] != " ":
                if board[row, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((row, c))
                break
            moves.append((row, c))
            c -= 1

        # Bishop-like movements
        # Top-left diagonal
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0:
            if board[r, c] != " ":
                if board[r, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r -= 1
            c -= 1

        # Bottom-right diagonal
        r, c = row + 1, col + 1
        while r <= 7 and c <= 7:
            if board[r, c] != " ":
                if board[r, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r += 1
            c += 1

        # Top-right diagonal
        r, c = row - 1, col + 1
        while r >= 0 and c <= 7:
            if board[r, c] != " ":
                if board[r, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r -= 1
            c += 1

        # Bottom-left diagonal
        r, c = row + 1, col - 1
        while r <= 7 and c >= 0:
            if board[r, c] != " ":
                if board[r, c] in ["R", "N", "B", "Q", "K", "P"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r += 1
            c -= 1

        return moves

    def move_rules_Q(self, position):
        row, col = position
        board = self.current_board
        moves = []

        # Rook-like movements
        # Upwards
        r = row - 1
        while r >= 0:
            if board[r, col] != " ":
                if board[r, col] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, col))
                break
            moves.append((r, col))
            r -= 1

        # Downwards
        r = row + 1
        while r <= 7:
            if board[r, col] != " ":
                if board[r, col] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, col))
                break
            moves.append((r, col))
            r += 1

        # Right
        c = col + 1
        while c <= 7:
            if board[row, c] != " ":
                if board[row, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((row, c))
                break
            moves.append((row, c))
            c += 1

        # Left
        c = col - 1
        while c >= 0:
            if board[row, c] != " ":
                if board[row, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((row, c))
                break
            moves.append((row, c))
            c -= 1

        # Bishop-like movements
        # Top-left diagonal
        r, c = row - 1, col - 1
        while r >= 0 and c >= 0:
            if board[r, c] != " ":
                if board[r, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r -= 1
            c -= 1

        # Bottom-right diagonal
        r, c = row + 1, col + 1
        while r <= 7 and c <= 7:
            if board[r, c] != " ":
                if board[r, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r += 1
            c += 1

        # Top-right diagonal
        r, c = row - 1, col + 1
        while r >= 0 and c <= 7:
            if board[r, c] != " ":
                if board[r, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r -= 1
            c += 1

        # Bottom-left diagonal
        r, c = row + 1, col - 1
        while r <= 7 and c >= 0:
            if board[r, c] != " ":
                if board[r, c] in ["r", "n", "b", "q", "k", "p"]:
                    moves.append((r, c))
                break
            moves.append((r, c))
            r += 1
            c -= 1

        return moves

    def possible_W_moves(self, threats=False):
        board = self.current_board
        rooks = {}
        knights = {}
        bishops = {}
        queens = {}
        pawns = {}

        # Rooks
        rook_positions = zip(*np.where(board == "R"))
        for pos in rook_positions:
            rooks[pos] = self.move_rules_R(pos)
        
        # Knights
        knight_positions = zip(*np.where(board == "N"))
        for pos in knight_positions:
            knights[pos] = self.move_rules_N(pos)
        
        # Bishops
        bishop_positions = zip(*np.where(board == "B"))
        for pos in bishop_positions:
            bishops[pos] = self.move_rules_B(pos)
        
        # Queens
        queen_positions = zip(*np.where(board == "Q"))
        for pos in queen_positions:
            queens[pos] = self.move_rules_Q(pos)
        
        # Pawns
        pawn_positions = zip(*np.where(board == "P"))
        for pos in pawn_positions:
            if not threats:
                pawns[pos], _ = self.move_rules_P(pos)
            else:
                _, pawns[pos] = self.move_rules_P(pos)
        
        components = {"R": rooks, "N": knights, "B": bishops, "Q": queens, "P": pawns}
        all_moves = list(itertools.chain.from_iterable(components.values()))
        
        return all_moves, components

    def move_rules_k(self):
        king_pos = np.where(self.current_board == "k")
        row, col = king_pos[0][0], king_pos[1][0]
        potential_moves = [
            (row + 1, col), (row - 1, col),
            (row, col + 1), (row, col - 1),
            (row + 1, col + 1), (row - 1, col - 1),
            (row + 1, col - 1), (row - 1, col + 1)
        ]
        valid_moves = []
        threats_list, _ = self.possible_W_moves(threats=True)

        for r, c in potential_moves:
            if 0 <= r <= 7 and 0 <= c <= 7:
                if self.current_board[r, c] in [" ", "Q", "B", "N", "P", "R"] and (r, c) not in threats_list:
                    valid_moves.append((r, c))
        
        # Castling options
        if self.castle("queenside") and not self.check_status():
            valid_moves.append((0, 2))
        if self.castle("kingside") and not self.check_status():
            valid_moves.append((0, 6))
        
        return valid_moves

    def possible_B_moves(self, threats=False):
        board = self.current_board
        rooks = {}
        knights = {}
        bishops = {}
        queens = {}
        pawns = {}

        # Rooks
        rook_positions = zip(*np.where(board == "r"))
        for pos in rook_positions:
            rooks[pos] = self.move_rules_r(pos)
        
        # Knights
        knight_positions = zip(*np.where(board == "n"))
        for pos in knight_positions:
            knights[pos] = self.move_rules_n(pos)
        
        # Bishops
        bishop_positions = zip(*np.where(board == "b"))
        for pos in bishop_positions:
            bishops[pos] = self.move_rules_b(pos)
        
        # Queens
        queen_positions = zip(*np.where(board == "q"))
        for pos in queen_positions:
            queens[pos] = self.move_rules_q(pos)
        
        # Pawns
        pawn_positions = zip(*np.where(board == "p"))
        for pos in pawn_positions:
            if not threats:
                pawns[pos], _ = self.move_rules_p(pos)
            else:
                _, pawns[pos] = self.move_rules_p(pos)
        
        components = {"r": rooks, "n": knights, "b": bishops, "q": queens, "p": pawns}
        all_moves = list(itertools.chain.from_iterable(components.values()))
        
        return all_moves, components

    def move_rules_K(self):
        king_pos = np.where(self.current_board == "K")
        row, col = king_pos[0][0], king_pos[1][0]
        potential_moves = [
            (row + 1, col), (row - 1, col),
            (row, col + 1), (row, col - 1),
            (row + 1, col + 1), (row - 1, col - 1),
            (row + 1, col - 1), (row - 1, col + 1)
        ]
        valid_moves = []
        threats_list, _ = self.possible_B_moves(threats=True)

        for r, c in potential_moves:
            if 0 <= r <= 7 and 0 <= c <= 7:
                if self.current_board[r, c] in [" ", "q", "b", "n", "p", "r"] and (r, c) not in threats_list:
                    valid_moves.append((r, c))
        
        # Castling options
        if self.castle("queenside") and not self.check_status():
            valid_moves.append((7, 2))
        if self.castle("kingside") and not self.check_status():
            valid_moves.append((7, 6))
        
        return valid_moves

    def move_piece(self, start, end, promotion="Q"):
        if self.player == 0:
            promoted = False
            s_row, s_col = start
            piece = self.current_board[s_row, s_col]
            self.current_board[s_row, s_col] = " "
            e_row, e_col = end

            # Update move counts for rooks and king
            if piece == "R":
                if start == (7, 0):
                    self.R1_move_count += 1
                elif start == (7, 7):
                    self.R2_move_count += 1
            if piece == "K":
                self.K_move_count += 1

            # Handle pawns
            if piece == "P":
                if abs(s_row - e_row) > 1:
                    self.en_passant = s_col
                    self.en_passant_move = self.move_count
                if abs(s_col - e_col) == 1 and self.current_board[e_row, e_col] == " ":
                    self.current_board[s_row, e_col] = " "
                if e_row == 0 and promotion in ["R", "B", "N", "Q"]:
                    self.current_board[e_row, e_col] = promotion
                    promoted = True
            if not promoted:
                self.current_board[e_row, e_col] = piece
            # Switch turn
            self.player = 1
            self.move_count += 1

        elif self.player == 1:
            promoted = False
            s_row, s_col = start
            piece = self.current_board[s_row, s_col]
            self.current_board[s_row, s_col] = " "
            e_row, e_col = end

            # Update move counts for rooks and king
            if piece == "r":
                if start == (0, 0):
                    self.r1_move_count += 1
                elif start == (0, 7):
                    self.r2_move_count += 1
            if piece == "k":
                self.k_move_count += 1

            # Handle pawns
            if piece == "p":
                if abs(s_row - e_row) > 1:
                    self.en_passant = s_col
                    self.en_passant_move = self.move_count
                if abs(s_col - e_col) == 1 and self.current_board[e_row, e_col] == " ":
                    self.current_board[s_row, e_col] = " "
                if e_row == 7 and promotion in ["r", "b", "n", "q"]:
                    self.current_board[e_row, e_col] = promotion
                    promoted = True
            if not promoted:
                self.current_board[e_row, e_col] = piece
            # Switch turn
            self.player = 0
            self.move_count += 1

        else:
            print("Invalid move:", start, end, promotion)

    def castle(self, side, inplace=False):
        if self.player == 0 and self.K_move_count == 0:
            if side == "queenside" and self.R1_move_count == 0 \
               and all(self.current_board[7, c] == " " for c in [1, 2, 3]):
                if inplace:
                    self.current_board[7, 0] = " "
                    self.current_board[7, 3] = "R"
                    self.current_board[7, 4] = " "
                    self.current_board[7, 2] = "K"
                    self.K_move_count += 1
                    self.player = 1
                return True
            elif side == "kingside" and self.R2_move_count == 0 \
                 and all(self.current_board[7, c] == " " for c in [5, 6]):
                if inplace:
                    self.current_board[7, 7] = " "
                    self.current_board[7, 5] = "R"
                    self.current_board[7, 4] = " "
                    self.current_board[7, 6] = "K"
                    self.K_move_count += 1
                    self.player = 1
                return True
        if self.player == 1 and self.k_move_count == 0:
            if side == "queenside" and self.r1_move_count == 0 \
               and all(self.current_board[0, c] == " " for c in [1, 2, 3]):
                if inplace:
                    self.current_board[0, 0] = " "
                    self.current_board[0, 3] = "r"
                    self.current_board[0, 4] = " "
                    self.current_board[0, 2] = "k"
                    self.k_move_count += 1
                    self.player = 0
                return True
            elif side == "kingside" and self.r2_move_count == 0 \
                 and all(self.current_board[0, c] == " " for c in [5, 6]):
                if inplace:
                    self.current_board[0, 7] = " "
                    self.current_board[0, 5] = "r"
                    self.current_board[0, 4] = " "
                    self.current_board[0, 6] = "k"
                    self.k_move_count += 1
                    self.player = 0
                return True
        return False

    def check_status(self):
        if self.player == 0:
            threats, _ = self.possible_B_moves(threats=True)
            king_pos = tuple(zip(*np.where(self.current_board == "K")))[0]
            if king_pos in threats:
                return True
        elif self.player == 1:
            threats, _ = self.possible_W_moves(threats=True)
            king_pos = tuple(zip(*np.where(self.current_board == "k")))[0]
            if king_pos in threats:
                return True
        return False

    def in_check_possible_moves(self):
        # Backup current state
        self.copy_board = self.current_board.copy()
        self.move_count_copy = self.move_count
        self.en_passant_copy = self.en_passant
        self.r1_move_count_copy = self.r1_move_count
        self.r2_move_count_copy = self.r2_move_count
        self.k_move_count_copy = self.k_move_count
        self.R1_move_count_copy = self.R1_move_count
        self.R2_move_count_copy = self.R2_move_count
        self.K_move_count_copy = self.K_move_count

        valid_moves = []
        if self.player == 0:
            _, components = self.possible_W_moves()
            king_pos = tuple(zip(*np.where(self.current_board == "K")))[0]
            components["K"] = {king_pos: self.move_rules_K()}
            for piece, moves_dict in components.items():
                for start_pos, ends in moves_dict.items():
                    for end_pos in ends:
                        self.move_piece(start_pos, end_pos)
                        self.player = 0
                        if not self.check_status():
                            valid_moves.append([start_pos, end_pos])
                        # Restore state
                        self.current_board = self.copy_board.copy()
                        self.en_passant = self.en_passant_copy
                        self.move_count = self.move_count_copy
                        self.r1_move_count = self.r1_move_count_copy
                        self.r2_move_count = self.r2_move_count_copy
                        self.k_move_count = self.k_move_count_copy
                        self.R1_move_count = self.R1_move_count_copy
                        self.R2_move_count = self.R2_move_count_copy
                        self.K_move_count = self.K_move_count_copy
            return valid_moves
        elif self.player == 1:
            _, components = self.possible_B_moves()
            king_pos = tuple(zip(*np.where(self.current_board == "k")))[0]
            components["k"] = {king_pos: self.move_rules_k()}
            for piece, moves_dict in components.items():
                for start_pos, ends in moves_dict.items():
                    for end_pos in ends:
                        self.move_piece(start_pos, end_pos)
                        self.player = 1
                        if not self.check_status():
                            valid_moves.append([start_pos, end_pos])
                        # Restore state
                        self.current_board = self.copy_board.copy()
                        self.en_passant = self.en_passant_copy
                        self.move_count = self.move_count_copy
                        self.r1_move_count = self.r1_move_count_copy
                        self.r2_move_count = self.r2_move_count_copy
                        self.k_move_count = self.k_move_count_copy
                        self.R1_move_count = self.R1_move_count_copy
                        self.R2_move_count = self.R2_move_count_copy
                        self.K_move_count = self.K_move_count_copy
            return valid_moves

    def actions(self):
        possible_actions = []
        if self.player == 0:
            _, components = self.possible_W_moves()
            king_pos = tuple(zip(*np.where(self.current_board == "K")))[0]
            components["K"] = {king_pos: self.move_rules_K()}
            for piece, moves_dict in components.items():
                for start_pos, ends in moves_dict.items():
                    for end_pos in ends:
                        if piece in ["P", "p"] and end_pos[0] in [0, 7]:
                            for promo in ["queen", "rook", "knight", "bishop"]:
                                possible_actions.append([start_pos, end_pos, promo])
                        else:
                            possible_actions.append([start_pos, end_pos, None])
            valid_actions = []
            for action in possible_actions:
                start, end, promo = action
                temp_board = copy.deepcopy(self)
                temp_board.move_piece(start, end, promo if promo else "Q")
                if not temp_board.check_status():
                    valid_actions.append(action)
            return valid_actions
        elif self.player == 1:
            _, components = self.possible_B_moves()
            king_pos = tuple(zip(*np.where(self.current_board == "k")))[0]
            components["k"] = {king_pos: self.move_rules_k()}
            for piece, moves_dict in components.items():
                for start_pos, ends in moves_dict.items():
                    for end_pos in ends:
                        if piece in ["P", "p"] and end_pos[0] in [0, 7]:
                            for promo in ["queen", "rook", "knight", "bishop"]:
                                possible_actions.append([start_pos, end_pos, promo])
                        else:
                            possible_actions.append([start_pos, end_pos, None])
            valid_actions = []
            for action in possible_actions:
                start, end, promo = action
                temp_board = copy.deepcopy(self)
                temp_board.move_piece(start, end, promo if promo else "Q")
                if not temp_board.check_status():
                    valid_actions.append(action)
            return valid_actions
