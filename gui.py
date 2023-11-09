import tkinter as tk
from main import ConnectFour
from tkinter import messagebox

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
        self.circles = []
        for r in range(self.game.rows):
            row_circles = []
            for c in range(self.game.columns):
                canvas = tk.Canvas(self.window, width=50, height=50, bg='white')
                canvas.grid(row=self.game.rows - 1 - r, column=c)
                # Draw an empty circle (white)
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
                    canvas.itemconfig(1, fill='blue')  # Update circle color
                elif piece == self.game.AI_PIECE:
                    canvas.itemconfig(1, fill='red')  # Update circle color
                else:
                    canvas.itemconfig(1, fill='white')  # Reset circle color

    def display_message(self, message):
        messagebox.showinfo("Game Info", message)

    def run(self):
        self.window.mainloop()

if __name__ == '__main__':
    game = ConnectFour()
    gui = ConnectFourGUI(game)
    gui.run()