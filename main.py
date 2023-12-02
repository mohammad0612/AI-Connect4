import numpy as np
import gym
from gym import spaces
import random
import sys
import math

class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ConnectFourEnv, self).__init__()
        self.action_space = spaces.Discrete(7)  # Assuming 7 columns
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.int32)  # Assuming 6x7 board
        self.game = ConnectFour()

    def step(self, action):
        # Check if the action is valid
        if not self.game.is_valid_location(self.game.board, action):
            return self.game.board, -10, True, {}  # Penalize and end game for invalid move

        # Drop the piece and check for game over or win
        row = self.game.get_next_open_row(self.game.board, action)
        self.game.drop_piece(self.game.board, row, action, self.game.AI_PIECE)

        if self.game.winning_move(self.game.board, self.game.AI_PIECE):
            # Calculate reward based on the number of moves taken
            # For example, a simple formula could be: 100 - number_of_moves
            # Where number_of_moves is the count of pieces on the board
            num_moves = np.count_nonzero(self.game.board)
            reward = max(10, 100 - num_moves)
            return self.game.board, reward, True, {}  # AI wins

        elif len(self.game.get_valid_locations(self.game.board)) == 0:
            return self.game.board, 0, True, {}  # Draw

        return self.game.board, 0, False, {}

    def reset(self):
        self.game.board = self.game.create_board()
        self.game.turn = 0  # Reset to player's turn
        return self.game.board

    def render(self, mode='human'):
        self.game.print_board(self.game.board)

    def close(self):
        pass


class ConnectFour:
    ## SET BOARD PARAMETERS HERE, CAN PLAY AROUND WITH THEM IN DEEP NEURAL NET POSSIBLY
    def __init__(self, rows=6, columns=7):
        self.rows = rows
        self.columns = columns
        self.USER = 0
        self.AI = 1
        self.EMPTY = 0
        self.USER_PIECE = 1
        self.AI_PIECE = 2
        self.WIN_RANGE = 4
        self.board = self.create_board()
        self.turn = random.choice([self.USER, self.AI])

    def create_board(self):
        board = np.zeros((self.rows, self.columns))
        return board

    def drop_piece(self, board, row, col, piece):
        if row is not None and 0 <= row < self.rows and 0 <= col < self.columns:
            board[row][col] = piece
        else:
            raise ValueError(f"Attempted to drop piece in an invalid location: row {row}, col {col}")

    def is_valid_location(self, board, col):
        return board[self.rows - 1][col] == self.EMPTY

    def get_next_open_row(self, board, col):
        for r in range(self.rows):
            if board[r][col] == self.EMPTY:
                return r

    def print_board(self, board):
        print(np.flip(board, 0))

    # check if any piece has won either player the game
    def winning_move(self, board, piece):
        for c in range(self.columns - self.WIN_RANGE + 1):
            for r in range(self.rows):
                if (board[r][c] == piece and board[r][c + 1] == piece and
                    board[r][c + 2] == piece and board[r][c + 3] == piece):
                    return True

        for c in range(self.columns):
            for r in range(self.rows - self.WIN_RANGE + 1):
                if (board[r][c] == piece and board[r + 1][c] == piece and
                    board[r + 2][c] == piece and board[r + 3][c] == piece):
                    return True

        for c in range(self.columns - self.WIN_RANGE + 1):
            for r in range(self.rows - self.WIN_RANGE + 1):
                if (board[r][c] == piece and board[r + 1][c + 1] == piece and
                    board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece):
                    return True

                if (board[r + 3][c] == piece and board[r + 3 - 1][c + 1] == piece and
                    board[r + 3 - 2][c + 2] == piece and board[r + 3 - 3][c + 3] == piece):
                    return True

        return False
    
    def is_game_over(self):
        if self.winning_move(self.board, self.USER_PIECE) or self.winning_move(self.board, self.AI_PIECE):
            return True
        if len(self.get_valid_locations(self.board)) == 0:
            return True
        return False
    
################################################################################    
##################### MINIMAX AND ALPHA BETA PRUNING ALGORITHMS ################
################################################################################

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = self.USER_PIECE
        if piece == self.USER_PIECE:
            opp_piece = self.AI_PIECE

        if window.count(piece) == self.WIN_RANGE:
            score += 100
        elif (window.count(piece) == self.WIN_RANGE - 1 and
              window.count(self.EMPTY) == 1):
            score += 5
        elif (window.count(piece) == self.WIN_RANGE - 2 and
              window.count(self.EMPTY) == 2):
            score += 2

        if (window.count(opp_piece) == self.WIN_RANGE - 1 and
            window.count(self.EMPTY) == 1):
            score -= 4

        return score

    def score_position(self, board, piece):
        score = 0

        center_array = [int(i) for i in list(board[:, self.columns // 2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        for r in range(self.rows):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(self.columns - self.WIN_RANGE + 1):
                window = row_array[c:c + self.WIN_RANGE]
                score += self.evaluate_window(window, piece)

        for c in range(self.columns):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(self.rows - self.WIN_RANGE + 1):
                window = col_array[r:r + self.WIN_RANGE]
                score += self.evaluate_window(window, piece)

        for r in range(self.rows - self.WIN_RANGE + 1):
            for c in range(self.columns - self.WIN_RANGE + 1):
                window = [board[r + i][c + i] for i in range(self.WIN_RANGE)]
                score += self.evaluate_window(window, piece)

        for r in range(self.rows - self.WIN_RANGE + 1):
            for c in range(self.columns - self.WIN_RANGE + 1):
                window = [board[r + self.WIN_RANGE - 1 - i][c + i] for i in range(self.WIN_RANGE)]
                score += self.evaluate_window(window, piece)

        return score

    def is_terminal_node(self, board):
        return (self.winning_move(board, self.USER_PIECE) or
                self.winning_move(board, self.AI_PIECE) or
                len(self.get_valid_locations(board)) == 0)

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board)
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.winning_move(board, self.AI_PIECE):
                    return (None, 100000000000000)
                elif self.winning_move(board, self.USER_PIECE):
                    return (None, -10000000000000)
                else:  # Game is over, no more valid moves
                    return (None, 0)
            else:  # Depth is zero
                return (None, self.score_position(board, self.AI_PIECE))
        if maximizingPlayer:
            value = -math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = np.copy(board)
                self.drop_piece(b_copy, row, col, self.AI_PIECE)
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else:  # Minimizing player
            value = math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = np.copy(board)
                self.drop_piece(b_copy, row, col, self.USER_PIECE)
                new_score = self.minimax(b_copy, depth - 1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(self.columns):
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations

    def pick_best_move(self, board, piece):
        valid_locations = self.get_valid_locations(board)
        best_score = -10000
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            temp_board = np.copy(board)
            self.drop_piece(temp_board, row, col, piece)
            score = self.score_position(temp_board, piece)
            if score > best_score:
                best_score = score
                best_col = col

        return best_col
    
################################################################################    
########################## PLAYING THE ACTUAL GAME #############################
################################################################################

    def draw_board(self, board):
        for r in range(self.rows):
            for c in range(self.columns):
                if board[r][c] == self.USER_PIECE:
                    print("P ", end="")
                elif board[r][c] == self.AI_PIECE:
                    print("AI ", end="")
                else:
                    print("- ", end="")
            print()
        print()

    def play_game(self):
        game_over = False

        while not game_over:
            self.print_board(self.board)

            if self.turn == self.USER:
                try:
                    col = int(input(f"Player 1, enter column (0-{self.columns - 1}): "))
                    if not self.is_valid_location(self.board, col):
                        print("Invalid move. Try again.")
                        continue
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    continue
            else:
                col, _ = self.minimax(self.board, 5, -sys.maxsize, sys.maxsize, True)

            row = self.get_next_open_row(self.board, col)
            self.drop_piece(self.board, row, col, self.turn + 1)

            if self.winning_move(self.board, self.turn + 1):
                self.print_board(self.board)
                print(f"Player {self.turn + 1} wins!!")
                game_over = True
            elif len(self.get_valid_locations(self.board)) == 0:
                print("It's a draw!")
                game_over = True
            print("-" * (self.columns * 4))  
            self.turn = 1 - self.turn

# if __name__ == '__main__':
#     connect = ConnectFour(rows=6, columns=7)
#     connect.play_game()

if __name__ == '__main__':
    env = ConnectFourEnv()
    env.reset()
    for _ in range(10):  # Example of playing 10 steps (actions are chosen randomly)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Game Over")
            break