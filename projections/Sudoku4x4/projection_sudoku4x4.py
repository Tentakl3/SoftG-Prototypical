import torch
import random

class Projection:
  def __init__(self):
        self.alpha = 0.0

  def swap_digits(self, board, d1, d2):
      new_board = board.copy()
      new_board[new_board == d1] = -1
      new_board[new_board == d2] = d1
      new_board[new_board == -1] = d2
      return new_board


  def swap_rows(self, board, r1, r2):
      new_board = board.copy()
      new_board[[r1, r2]] = new_board[[r2, r1]]
      return new_board


  def swap_cols(self, board, c1, c2):
      new_board = board.copy()
      new_board[:, [c1, c2]] = new_board[:, [c2, c1]]
      return new_board

  def mutate_sudoku_4x4_neg(self, board):
      with torch.no_grad():

        new_board = board.detach().cpu().numpy().copy()
        new_boards = []

        for d1 in range(1, 5):
          for d2 in range(1, 5):
            if d1 != d2:
              new_boards.append(self.swap_digits(new_board, d1, d2))

      return new_boards    

  def mutate_sudoku_4x4(self, board):
      """
      Generate a mutated valid Sudoku solution from a 4x4 solution.
      """
      with torch.no_grad():
        new_board = board.detach().cpu().numpy().copy()
        new_boards = []
        
        for d1 in range(1, 5):
          for d2 in range(1, 5):
            if d1 != d2:
              new_boards.append(self.swap_digits(new_board, d1, d2))


        band = random.choice([0, 2])
        r1, r2 = band, band + 1
        new_boards.append(self.swap_rows(new_board, r1, r2))
    
        stack = random.choice([0, 2])
        c1, c2 = stack, stack + 1
        new_boards.append(self.swap_cols(new_board, c1, c2))

        new_boards.append(self.swap_rows(new_board, 0, 2))
        new_boards.append(self.swap_rows(new_board, 1, 3))

        new_boards.append(self.swap_cols(new_board, 0, 2))
        new_boards.append(self.swap_cols(new_board, 1, 3))

      return new_boards