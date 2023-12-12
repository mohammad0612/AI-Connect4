import numpy as np
import pandas as pd
import sys
from main import ConnectFour  # Assuming this is your ConnectFour class
import csv

class Connect4DatasetGenerator(ConnectFour):
    def __init__(self):
        super().__init__()

    def play_game(self, seen_states):
        game_over = False
        game_states = []
        best_moves = []

        while not game_over:
            game_state_str = str(np.flip(self.board, 0))
            best_move, _ = self.minimax(self.board, 5, -sys.maxsize, sys.maxsize, True)
            if game_state_str not in seen_states:
                seen_states.add(game_state_str)
                game_states.append(np.flip(self.board, 0).copy().flatten())
                best_moves.append(best_move)

            row = self.get_next_open_row(self.board, best_move)
            self.drop_piece(self.board, row, best_move, self.turn + 1)

            if self.winning_move(self.board, self.turn + 1):
                game_over = True
            elif len(self.get_valid_locations(self.board)) == 0:
                game_over = True
            self.turn = self.USER if self.turn == self.AI else self.AI

        return game_states, best_moves

# Load the existing game states from the CSV file into a set
df = pd.read_csv('connect4_data.csv')
existing_game_states = set(df['Game State'])

seen_states = set()
game_states = []
best_moves = []
for i in range(5):  # Adjust as needed
    print("Playing game", i)
    game = Connect4DatasetGenerator()
    states, moves = game.play_game(seen_states)
    print(f"Recorded {len(states)} states and {len(moves)} moves")
    game_states.extend(states)
    best_moves.extend(moves)

print(f"Total states: {len(game_states)}, total moves: {len(best_moves)}")

# Assume game_states and best_moves are your lists of game states and moves
with open('connect4_data.csv', 'a', newline='') as f:  # Open the file in append mode
    writer = csv.writer(f)
    for state, move in zip(game_states, best_moves):
        if str(state) not in existing_game_states:  # Only write the game state if it's not already in the file
          writer.writerow([','.join(map(str, state)), move])

print(f"Total states: {len(game_states)}, total moves: {len(best_moves)}")