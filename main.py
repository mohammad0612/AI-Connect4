import random
import math 
import copy

class ConnectFour:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = [['-' for _ in range(cols)] for _ in range(rows)]
        self.turn = 'X'  # Player X goes first

    ##########################################################################
    # CREATION OF BOARD AND GAME PLAY MECHANICS 
    ##########################################################################
    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        print(' '.join(map(str, range(self.cols))))

    def drop_piece(self, col):
        if col < 0 or col >= self.cols or self.board[0][col] != '-':
            return False
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == '-':
                self.board[row][col] = self.turn
                break
        return True

    def check_winner(self):
        # Horizontal, vertical and diagonal checks
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if self.board[row][col] == self.turn and all(self.board[row][col + i] == self.turn for i in range(4)):
                    return True

        for col in range(self.cols):
            for row in range(self.rows - 3):
                if self.board[row][col] == self.turn and all(self.board[row + i][col] == self.turn for i in range(4)):
                    return True

        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if self.board[row][col] == self.turn and all(self.board[row + i][col + i] == self.turn for i in range(4)):
                    return True

                if self.board[row + 3][col] == self.turn and all(self.board[row + 3 - i][col + i] == self.turn for i in range(4)):
                    return True
        return False
    
    def play_game(self, bot_play=True):
        while True:
            self.print_board()
            if self.turn == 'X':
                if bot_play:
                    col, minimax_score = self.minimax(5, -math.inf, math.inf, True)
                    if col is None:
                        print("No valid moves left!")
                        break
                else:
                    col = int(input(f"Player {self.turn}, choose column (0-{self.cols - 1}): "))
            else:
                col = int(input(f"Player {self.turn}, choose column (0-{self.cols - 1}): "))

            if self.drop_piece(col):
                if self.check_winner():
                    self.print_board()
                    print(f"Player {self.turn} wins!")
                    break
                if self.is_full():
                    self.print_board()
                    print("It's a draw!")
                    break
                self.switch_turn()
            else:
                print("Invalid move, try again.")

    ##########################################################################
    # MINIMAX ALGO AND BOARD STATE EVAL
    ##########################################################################
    def is_full(self):
        return all(self.board[0][col] != '-' for col in range(self.cols))

    def switch_turn(self):
        self.turn = 'O' if self.turn == 'X' else 'X'

    def get_valid_locations(self):
        return [col for col in range(self.cols) if self.board[0][col] == '-']

    def is_terminal_node(self):
        return self.check_winner() or self.is_full()
    
    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = 'O' if piece == 'X' else 'X'
        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count('-') == 1:
            score += 5
        elif window.count(piece) == 2 and window.count('-') == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count('-') == 1:
            score -= 4

        return score

    def evaluate_board(self, piece):
        score = 0

        ## Score center column
        center_array = [self.board[i][self.cols//2] for i in range(self.rows)]
        center_count = center_array.count(piece)
        score += center_count * 3

        ## Score Horizontal
        for r in range(self.rows):
            row_array = [self.board[r][c] for c in range(self.cols)]
            for c in range(self.cols - 3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, piece)

        ## Score Vertical
        for c in range(self.cols):
            col_array = [self.board[r][c] for r in range(self.rows)]
            for r in range(self.rows - 3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, piece)

        ## Score positive sloped diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [self.board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        ## Score negative sloped diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [self.board[r+3-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        return score

    
    def minimax(self, depth, alpha, beta, maximizingPlayer):
        valid_locations = self.get_valid_locations()
        is_terminal = self.is_terminal_node()
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_winner():
                    # If the current player is maximizing, then they have won
                    return (None, float('inf') if maximizingPlayer else float('-inf'))
                else: # Game is over, no more valid moves
                    return (None, 0)
            else: # Depth is zero
                return (None, self.evaluate_board('X' if maximizingPlayer else 'O'))
        if maximizingPlayer:
            value = float('-inf')
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(col)
                b_copy = copy.deepcopy(self.board)
                b_copy[row][col] = 'X'
                new_score = self.minimax(depth-1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value
        else: # Minimizing player
            value = float('inf')
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(col)
                b_copy = copy.deepcopy(self.board)
                b_copy[row][col] = 'O'
                new_score = self.minimax(depth-1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def get_next_open_row(self, col):
        for r in range(self.rows-1, -1, -1):
            if self.board[r][col] == '-':
                return r
        raise Exception("Column is full")
            


if __name__ == '__main__':
    game = ConnectFour()
    game.play_game()
