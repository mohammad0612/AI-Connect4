import numpy as np
from main import ConnectFour
from tensorflow.keras.models import load_model

class CNNAI:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        
    def predict_move(self, board_state):
        board_state = np.array(board_state).reshape((1, 6, 7, 1))
        probabilities = self.model.predict(board_state)[0]
        return np.argmax(probabilities)

def play_game(cnn_ai, minimax_ai, game_instance):
    while not game_instance.is_game_over():
        if game_instance.turn == game_instance.USER:
            # Choose a valid move from CNN AI
            valid_move = False
            attempted_columns = set()
            while not valid_move:
                col = cnn_ai.predict_move(game_instance.board)
                if col in attempted_columns:
                    print(f"CNN AI is attempting to play in a full column: {col}")
                    # Select a random column that hasn't been attempted yet
                    col = np.random.choice([c for c in range(game_instance.columns) if c not in attempted_columns])
                
                if game_instance.is_valid_location(game_instance.board, col):
                    valid_move = True
                else:
                    attempted_columns.add(col)

                if len(attempted_columns) == game_instance.columns:
                    print("All columns are full. It's a draw.")
                    return "Draw"
        else:
            # Choose a valid move from Minimax AI
            valid_move = False
            while not valid_move:
                col, _ = minimax_ai.minimax(game_instance.board, 5, -np.inf, np.inf, True)
                if game_instance.is_valid_location(game_instance.board, col):
                    valid_move = True
                else:
                    print(f"Minimax AI is attempting to play in a full column: {col}")
                    # This should not happen if Minimax AI is implemented correctly

        row = game_instance.get_next_open_row(game_instance.board, col)
        game_instance.drop_piece(game_instance.board, row, col, game_instance.turn + 1)

        if game_instance.winning_move(game_instance.board, game_instance.turn + 1):
            return 'CNN' if game_instance.turn == game_instance.USER else 'Minimax'

        game_instance.turn = 1 - game_instance.turn  # Switch turns

    return "Draw"




def main():
    cnn_ai = CNNAI('connect_four_dqn.h5')  # Update this path
    minimax_ai = ConnectFour()
    
    results = {"CNN": 0, "Minimax": 0, "Draw": 0}
    num_games = 10

    for i in range(num_games):
        print(i)
        game_instance = ConnectFour()
        winner = play_game(cnn_ai, minimax_ai, game_instance)
        results[winner] += 1

    # Print the results
    print(f"After {num_games} games, we have:")
    print(f"CNN AI wins: {results['CNN']} ({(results['CNN'] / num_games) * 100}%)")
    print(f"Minimax AI wins: {results['Minimax']} ({(results['Minimax'] / num_games) * 100}%)")
    print(f"Draws: {results['Draw']} ({(results['Draw'] / num_games) * 100}%)")

if __name__ == '__main__':
    main()
