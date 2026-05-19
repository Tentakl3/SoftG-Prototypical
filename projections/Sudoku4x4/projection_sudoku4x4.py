import torch
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Projection:
  def __init__(self):
        self.alpha = 0.0

  def swap_digits(self, board, d1, d2):
      new_board = board.copy()
      new_board[new_board == d1] = -1
      new_board[new_board == d2] = d1
      new_board[new_board == -1] = d2
      return new_board

  def tensorized_swap_digits(self, boards, d1, d2):
      new_boards = boards.clone()  # [B, K, 4, 4]
      d1_expand = d1.unsqueeze(-1).unsqueeze(-1)
      d2_expand = d2.unsqueeze(-1).unsqueeze(-1)

      fill_value = torch.tensor(-1, device=new_boards.device, dtype=new_boards.dtype)
      new_boards = torch.where(new_boards == d1_expand, fill_value, new_boards)
      new_boards = torch.where(new_boards == d2_expand, d1_expand, new_boards)
      new_boards = torch.where(new_boards == fill_value, d2_expand, new_boards)
      return new_boards
  
  def swap_rows(self, board, r1, r2):
      new_board = board.copy()
      new_board[[r1, r2]] = new_board[[r2, r1]]
      return new_board

  def tensorized_swap_rows(self, boards, r1, r2):
      new_boards = boards.clone() #[B, K, 4, 4]
      new_boards[:, [r1, r2]] = new_boards[:, [r2, r1]]
      return new_boards

  def swap_cols(self, board, c1, c2):
      new_board = board.copy()
      new_board[:, [c1, c2]] = new_board[:, [c2, c1]]
      return new_board

  def tensorized_swap_cols(self, boards, c1, c2):
      new_boards = boards.clone() #[B, K, 4, 4]
      new_boards[:, :, [c1, c2]] = new_boards[:, :, [c2, c1]]
      return new_boards

  def tensorized_mutate_sudoku_4x4(self, boards):
      B, K, _, _ = boards.shape
      digits = torch.tensor([1, 2, 3, 4]).to(device)
      indices = torch.randint(0, 4, (B, K, 2)).to(device) # [B, K, 2]
      d1 = digits[indices[:, :, 0]]  # [B, K]
      d2 = digits[indices[:, :, 1]]  # [B, K]

      mutated_boards = self.tensorized_swap_digits(boards, d1, d2)

      return mutated_boards.to(device)

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