# Assuming LearningAI is in a separate file, let's import it.
from learning_ai import LearningAI
from main import ConnectFour
import numpy as np
import math

def play_game(learning_ai, minimax_ai, game_instance):
    while not game_instance.is_game_over():
        if game_instance.turn == game_instance.USER:
            # Choose a valid move from Learning AI
            learning_ai.move(game_instance)  # Predicts the next move
            col = learning_ai.choice
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
            return 'LearningAI' if game_instance.turn == game_instance.USER else 'Minimax'

        game_instance.turn = 1 - game_instance.turn  # Switch turns

    return "Draw"

def main():
    learning_ai = LearningAI('connect4_next_move_model.h5', p=1, name='Paul')  # Update the model path
    minimax_ai = ConnectFour()
    
    results = {"LearningAI": 0, "Minimax": 0, "Draw": 0}
    num_games = 10

    for i in range(num_games):
        print(i)
        game_instance = ConnectFour()
        winner = play_game(learning_ai, minimax_ai, game_instance)
        results[winner] += 1

    # Print the results
    print(f"After {num_games} games, we have:")
    print(f"LearningAI wins: {results['LearningAI']} ({(results['LearningAI'] / num_games) * 100}%)")
    print(f"Minimax AI wins: {results['Minimax']} ({(results['Minimax'] / num_games) * 100}%)")
    print(f"Draws: {results['Draw']} ({(results['Draw'] / num_games) * 100}%)")

if __name__ == '__main__':
    main()
