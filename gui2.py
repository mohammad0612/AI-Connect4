import tkinter as tk
from main import ConnectFour
from tkinter import messagebox
import torch
import numpy as np

# Import the DQN class
from DQN import DQN

class ConnectFourGUI:
    def __init__(self, game):
        self.game = game
        self.window = tk.Tk()
        self.window.title("Connect Four")
        self.create_board()
        self.load_dqn_agent()

    def load_dqn_agent(self):
        # Load the trained DQN model
        input_size = 42  # 6 rows * 7 columns
        output_size = 7  # 7 possible actions (columns)
        sizes = [512,256,128,64,64]
        self.dqn_agent = DQN(input_size, output_size, sizes)
        self.dqn_agent.load("connect_four_dqn_agent2.pth", device=torch.device("cpu"))

    def on_click(self, col):
        # Player's turn
        if self.game.is_valid_location(self.game.board, col):
            row = self.game.get_next_open_row(self.game.board, col)
            self.game.drop_piece(self.game.board, row, col, self.game.USER_PIECE)
            self.update_board()

            if self.game.winning_move(self.game.board, self.game.USER_PIECE):
                self.display_message("Player Wins!")
                return

            # AI's turn using DQN agent
            with torch.no_grad():
                state = torch.tensor(self.game.board.flatten(), dtype=torch.float32)
                q_values = self.dqn_agent(state)
                ai_col = q_values.argmax().item()

            if self.game.is_valid_location(self.game.board, ai_col):
                ai_row = self.game.get_next_open_row(self.game.board, ai_col)
                self.game.drop_piece(self.game.board, ai_row, ai_col, self.game.AI_PIECE)
                self.update_board()

                if self.game.winning_move(self.game.board, self.game.AI_PIECE):
                    self.display_message("AI Wins!")
                    return

            self.update_board()

    def create_board(self):
        self.circles = []
        for r in range(self.game.rows):
            row_circles = []
            for c in range(self.game.columns):
                canvas = tk.Canvas(self.window, width=50, height=50, bg='white')
                canvas.grid(row=self.game.rows - 1 - r, column=c)
                canvas.create_oval(10, 10, 40, 40, fill='white', outline='black')
                canvas.bind("<Button-1>", lambda event, col=c: self.on_click(col))
                row_circles.append(canvas)
            self.circles.append(row_circles)
        self.circles.reverse()

    def update_board(self):
        for r in range(self.game.rows):
            for c in range(self.game.columns):
                gui_row = self.game.rows - 1 - r
                piece = self.game.board[r][c]
                canvas = self.circles[gui_row][c]
                if piece == self.game.USER_PIECE:
                    canvas.itemconfig(1, fill='blue')
                elif piece == self.game.AI_PIECE:
                    canvas.itemconfig(1, fill='red')
                else:
                    canvas.itemconfig(1, fill='white')

    def display_message(self, message):
        messagebox.showinfo("Game Info", message)

    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    game = ConnectFour()
    gui = ConnectFourGUI(game)
    gui.run()
