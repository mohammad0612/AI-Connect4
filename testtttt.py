#!/usr/bin/env python3

from board import Board
from players import LearningAI
from players import RandomAI
from players import HumanPlayer
import numpy as np

def play_connect4():
    # Create a Connect 4 board
    connect_four_board = Board()

    # Create a LearningAI player
    learning_ai = LearningAI('connect4_model.h5')  # Replace 'None' with the Keras model path
    human_ai = HumanPlayer('Momo')

    # Play until there's a winner or no more moves left
    while connect_four_board.N_moves_left > 0:
        # Display the current board
        connect_four_board.display_grid()

        # User's turn
        col = int(input(f"Player 1, enter column (0-6)"))
        human_ai.move(connect_four_board, col)
        connect_four_board.update(human_ai)  # Assuming user is Player 1

        # Check if the user wins
        if connect_four_board.check_vectors(learning_ai):
            connect_four_board.display_grid()
            print("Congratulations! You won!")
            break

        # AI's turn
        learning_ai.move(connect_four_board)
        connect_four_board.update(learning_ai)

        # Check if the AI wins
        if connect_four_board.check_vectors(learning_ai):
            connect_four_board.display_grid()
            print("Sorry, you lost! The AI won.")
            break

    # Check for a tie
    if connect_four_board.N_moves_left == 0:
        connect_four_board.display_grid()
        print("It's a tie!")

# Play Connect 4
play_connect4()
