import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from ai_comp import CNNAI
import random
import sys
import math
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('connect4_demo.h5')

# Define the DQN agent
class DQNAgent:
    def __init__(self, model):
        self.model = model
        self.target_model = tf.keras.models.clone_model(model)
        self.target_model.set_weights(model.get_weights())
        self.optimizer = Adam(learning_rate=0.001)
        self.loss_fn = MeanSquaredError()

    def train(self, states, actions, rewards, next_states, dones, batch_size=32, gamma=0.99):
        # Compute Q-values for current states
        q_values = self.model.predict(states)

        # Compute Q-values for next states using the target model
        next_q_values = self.target_model.predict(next_states)

        # Compute target Q-values
        targets = np.copy(q_values)
        targets[np.arange(batch_size), actions] = rewards + gamma * np.max(next_q_values, axis=1) * (1 - dones)

        # Train the model
        with tf.GradientTape() as tape:
            q_values_pred = self.model(states, training=True)
            loss = self.loss_fn(targets, q_values_pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

# Your ConnectFour game logic (if you have it in a separate file)
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

# Define your gameplay loop and update the agent
def play_game(agent, minimax_ai, game_instance):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    while not game_instance.is_game_over():
        if game_instance.turn == game_instance.USER:
            # Choose a valid move from DQN AI
            state = np.array(game_instance.board).reshape((1, 6, 7, 1))
            action = agent.model.predict(state)[0]
            print("State shape:", state.shape)
            print("Action shape:", action.shape)
            col = np.argmax(action)
            while not game_instance.is_valid_location(game_instance.board, col):
                # Choose the next best action if the selected column is invalid
                action[col] = -np.inf
                col = np.argmax(action)

        else:
            # Choose a valid move from Minimax AI
            valid_move = False
            while not valid_move:
                col, _ = minimax_ai.minimax(game_instance.board, 5, -np.inf, np.inf, True)
                if game_instance.is_valid_location(game_instance.board, col):
                    valid_move = True

        row = game_instance.get_next_open_row(game_instance.board, col)
        game_instance.drop_piece(game_instance.board, row, col, game_instance.turn + 1)

        if game_instance.winning_move(game_instance.board, game_instance.turn + 1):
            return states, actions, rewards, next_states, dones

        game_instance.turn = 1 - game_instance.turn  # Switch turns

    return states, actions, rewards, next_states, dones

def should_update_target_model(game_counter):
    # Fill in the logic to decide when to update the target model
    # For example, update every 'n' games
    return game_counter % 10 == 0  # Update every 10 games as an example

def should_stop_gameplay(game_counter):
  # Fill in the logic to stop the gameplay loop
  # For instance, stop after a certain number of games
  return game_counter >= 1000  # Stop after 1000 games as an example

def main():
    cnn_ai = CNNAI('connect4_demo.h5')  # Update this path
    minimax_ai = ConnectFour()

    agent = DQNAgent(model)

    game_counter = 0

    while True:
        game_instance = ConnectFour()
        states, actions, rewards, next_states, dones = play_game(agent, minimax_ai, game_instance)
        agent.train(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

        game_counter += 1

        if should_update_target_model(game_counter):
            agent.update_target_model()

        if should_stop_gameplay(game_counter):
            break

    agent.update_target_model()
    model.save('/Users/momo/Documents/Git Projects/AI-Connect4/optimized_model.h5')

if __name__ == '__main__':
    main()
