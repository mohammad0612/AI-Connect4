import random 
from main import ConnectFour, ConnectFourEnv

def find_blocking_move(board, player_piece):
    opponent_piece = 2 if player_piece == 1 else 1
    connect_four = ConnectFour()  # Create an instance of ConnectFour class

    for c in range(7):
        for r in range(6):
            if board[r][c] == 0:
                # Temporarily drop the opponent's piece
                board[r][c] = opponent_piece
                if connect_four.winning_move(board, opponent_piece):
                    board[r][c] = 0  # Undo the move
                    return c
                board[r][c] = 0  # Undo the move

    return None

def bot_move(board, player_piece):
    block_move = find_blocking_move(board, player_piece)
    if block_move is not None:
        return block_move
    else:
        return random_valid_move(board)
    
def random_valid_move(board):
    valid_locations = [c for c in range(7) if board[5][c] == 0]
    return random.choice(valid_locations) if valid_locations else None