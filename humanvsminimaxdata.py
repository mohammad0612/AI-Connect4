import numpy as np
import csv
import sys
from main import ConnectFour

class Connect4DatasetGenerator(ConnectFour):
    def __init__(self):
        super().__init__()

    def play_game(self, seen_states):
        game_over = False
        game_states = []
        best_moves = []

        while not game_over:
            game_state_str = str(np.flip(self.board, 0).flatten().tolist())
            print(np.flip(self.board, 0))

            if self.turn == self.AI:
                best_move, _ = self.minimax(self.board, 5, -sys.maxsize, sys.maxsize, True)
                row = self.get_next_open_row(self.board, best_move)
                self.drop_piece(self.board, row, best_move, self.AI_PIECE)

                if game_state_str not in seen_states:
                    seen_states.add(game_state_str)
                    game_states.append(np.flip(self.board, 0).flatten())
                    best_moves.append(best_move)
            else:
                user_move = int(input("Enter your move: 0-6"))
                row = self.get_next_open_row(self.board, user_move)
                self.drop_piece(self.board, row, user_move, self.USER_PIECE)

            if self.winning_move(self.board, self.turn + 1):
                game_over = True
            elif len(self.get_valid_locations(self.board)) == 0:
                game_over = True
            self.turn = self.USER if self.turn == self.AI else self.AI

        return game_states, best_moves


game_states = []
best_moves = []
seen_states = set()

with open('connect4_data.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    for row in reader:
        seen_states.add(row[0])

for i in range(1):  # Adjust as needed
    print("Playing game", i)
    game = Connect4DatasetGenerator()
    states, moves = game.play_game(seen_states)
    print(f"Recorded {len(states)} states and {len(moves)} moves")
    game_states.extend(states)
    best_moves.extend(moves)

print(f"Total states: {len(game_states)}, total moves: {len(best_moves)}")

with open('connect4_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for state, move in zip(game_states, best_moves):
        writer.writerow([','.join(map(str, state)), move])

print(f"Total states: {len(game_states)}, total moves: {len(best_moves)}")