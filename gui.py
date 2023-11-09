import tkinter as tk
from main import ConnectFour
from tkinter import messagebox
import numpy as np

class ConnectFourGUI:
    def __init__(self, game):
        self.game = game
        self.window = tk.Tk()
        self.window.title("Connect Four")
        self.create_board()

    def on_click(self, col):
        # Check if the column is valid for a move
        if self.game.is_valid_location(self.game.board, col):
            # Get the next open row in the chosen column
            row = self.game.get_next_open_row(self.game.board, col)
            # Drop the piece in the board
            self.game.drop_piece(self.game.board, row, col, self.game.USER_PIECE)
            self.update_board()

            # Check for a winning move
            if self.game.winning_move(self.game.board, self.game.USER_PIECE):
                self.display_message("Player Wins!")
                return  # End the game or reset

            # AI's turn
            ai_col, _ = self.game.minimax(self.game.board, 5, -float("inf"), float("inf"), True)
            if self.game.is_valid_location(self.game.board, ai_col):
                ai_row = self.game.get_next_open_row(self.game.board, ai_col)
                self.game.drop_piece(self.game.board, ai_row, ai_col, self.game.AI_PIECE)
                self.update_board()

                if self.game.winning_move(self.game.board, self.game.AI_PIECE):
                    self.display_message("AI Wins!")
                    return  # End the game or reset

            # Redraw the board after each move
            self.update_board()

    def create_board(self):
        self.buttons = []
        for r in range(self.game.rows):
            row_buttons = []
            for c in range(self.game.columns):
                btn = tk.Button(self.window, text=' ', command=lambda col=c: self.on_click(col), height=2, width=4)
                # Add buttons starting from the bottom row
                gui_row = self.game.rows - 1 - r
                btn.grid(row=gui_row, column=c)
                row_buttons.append(btn)
            self.buttons.append(row_buttons)
        self.buttons.reverse()

    def update_board(self):
        for r in range(self.game.rows):
            for c in range(self.game.columns):
                # Calculate the GUI row from the bottom up
                gui_row = self.game.rows - 1 - r
                # Get the backend board value
                piece = self.game.board[r][c]
                btn = self.buttons[gui_row][c]
                if piece == self.game.USER_PIECE:
                    btn.config(text="P", bg="blue")
                elif piece == self.game.AI_PIECE:
                    btn.config(text="AI", bg="red")
                else:
                    btn.config(text=" ", bg="white")

    def display_message(self, message):
        messagebox.showinfo("Game Info", message)

    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    game = ConnectFour()
    gui = ConnectFourGUI(game)
    gui.run()