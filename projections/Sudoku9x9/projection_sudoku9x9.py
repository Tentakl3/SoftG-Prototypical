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
      new_boards = boards.clone()  # [B, K, 9, 9]
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
      new_boards = boards.clone() #[B, K, 9, 9]
      new_boards[:, [r1, r2]] = new_boards[:, [r2, r1]]
      return new_boards

  def swap_cols(self, board, c1, c2):
      new_board = board.copy()
      new_board[:, [c1, c2]] = new_board[:, [c2, c1]]
      return new_board

  def tensorized_swap_cols(self, boards, c1, c2):
      new_boards = boards.clone() #[B, K, 9, 9]
      new_boards[:, :, [c1, c2]] = new_boards[:, :, [c2, c1]]
      return new_boards

  def tensorized_mutate_sudoku_9x9(self, boards):
      B, K, _, _ = boards.shape
      digits = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).to(device)
      indices = torch.randint(0, 9, (B, K, 2)).to(device) # [B, K, 2]
      d1 = digits[indices[:, :, 0]]  # [B, K]
      d2 = digits[indices[:, :, 1]]  # [B, K]

      mutated_boards = self.tensorized_swap_digits(boards, d1, d2)

      return mutated_boards.to(device)

  def mutate_sudoku_9x9_neg(self, board):
      with torch.no_grad():

        new_board = board.detach().cpu().numpy().copy()
        new_boards = []

        for d1 in range(1, 10):
          for d2 in range(1, 10):
            if d1 != d2:
              new_boards.append(self.swap_digits(new_board, d1, d2))

      return new_boards

  def mutate_sudoku_9x9(self, board):
      """
      Generate a mutated valid Sudoku solution from a 9x9 solution.
      """
      with torch.no_grad():
        new_board = board.detach().cpu().numpy().copy()
        new_boards = []

        for d1 in range(1, 10):
          for d2 in range(1, 10):
            if d1 != d2:
              new_boards.append(self.swap_digits(new_board, d1, d2))

        # In-band row swaps (within each 3-row band).
        band = random.choice([0, 3, 6])
        offsets = random.sample([0, 1, 2], 2)
        r1, r2 = band + offsets[0], band + offsets[1]
        new_boards.append(self.swap_rows(new_board, r1, r2))

        # In-stack col swaps (within each 3-col stack).
        stack = random.choice([0, 3, 6])
        offsets = random.sample([0, 1, 2], 2)
        c1, c2 = stack + offsets[0], stack + offsets[1]
        new_boards.append(self.swap_cols(new_board, c1, c2))

        # Band swaps (whole 3-row bands).
        band_a, band_b = random.sample([0, 3, 6], 2)
        for di in range(3):
            tmp_board = new_board.copy()
            tmp_board[[band_a + di, band_b + di]] = tmp_board[[band_b + di, band_a + di]]
            new_boards.append(tmp_board)

        # Stack swaps (whole 3-col stacks).
        stack_a, stack_b = random.sample([0, 3, 6], 2)
        for dj in range(3):
            tmp_board = new_board.copy()
            tmp_board[:, [stack_a + dj, stack_b + dj]] = tmp_board[:, [stack_b + dj, stack_a + dj]]
            new_boards.append(tmp_board)

      return new_boards
