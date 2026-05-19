from z3 import *
import torch
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sampler:
    def __init__(self):
        self.sat_boards_cache = torch.tensor(self.gen_sat_boards(), dtype=torch.long).to(device)
        self.sat_boards_cache_array = self.sat_boards_cache.cpu().tolist()  # For quick checking of unsat samples

    def sudoku_4x4_solver(self):
        s = Solver()
        X = [[Int(f"x_{i}_{j}") for j in range(4)] for i in range(4)]

        # Domain constraints
        for i in range(4):
            for j in range(4):
                s.add(And(X[i][j] >= 1, X[i][j] <= 4))

        # Row and column constraints
        for i in range(4):
            s.add(Distinct(X[i]))
            s.add(Distinct([X[j][i] for j in range(4)]))

        # 2x2 subgrids
        for i in [0, 2]:
            for j in [0, 2]:
                block = [X[i + di][j + dj] for di in range(2) for dj in range(2)]
                s.add(Distinct(block))

        return s, X

    def gen_sat_boards(self):
        s, X = self.sudoku_4x4_solver()
        solutions = []

        while s.check() == sat:
            m = s.model()

            board = np.array(
                [[m.evaluate(X[i][j]).as_long() for j in range(4)] for i in range(4)] #type:ignore
            )
            solutions.append(board)

            block = Or([
                X[i][j] != m.evaluate(X[i][j])
                for i in range(4)
                for j in range(4)
            ])
            s.add(block)

        solutions = np.array(solutions)  # Shape: [num_solutions, 4, 4]
        return solutions

    def generate_unsat_boards_set(self, sat_boards):
        neg_solutions = []
        for board in sat_boards:
            correct_board = board.copy()
            corrupted_board = self.generate_unsat_board(correct_board)
            check = self.check_sudoku_4x4(board)
            while check:
                correct_board = board.copy()
                corrupted_board = self.generate_unsat_board(correct_board)
                check = self.check_sudoku_4x4(corrupted_board)
            neg_solutions.append(corrupted_board)

        return neg_solutions
        
    def generate_unsat_board(self, board):
        violation_type = random.choice(["row", "col", "block"])

        if violation_type == "row":
            i = random.randint(0, 3)
            board[i][1] = board[i][0]

        elif violation_type == "col":
            j = random.randint(0, 3)
            board[1][j] = board[0][j]

        else:  # block
            bi = random.choice([0, 2])
            bj = random.choice([0, 2])
            board[bi][bj + 1] = board[bi][bj]

        return board

    def check_sudoku_4x4(self, board):
        s = Solver()

        # Z3 variables
        X = [[Int(f"x_{i}_{j}") for j in range(4)] for i in range(4)]

        # Cell constraints
        for i in range(4):
            for j in range(4):
                s.add(X[i][j] >= 1, X[i][j] <= 4)
                s.add(X[i][j] == board[i][j])

        for i in range(4):
            s.add(Distinct(X[i]))

        for j in range(4):
            s.add(Distinct([X[i][j] for i in range(4)]))

        for i in range(0, 4, 2):
            for j in range(0, 4, 2):
                block = [
                    X[i + di][j + dj]
                    for di in range(2)
                    for dj in range(2)
                ]
                s.add(Distinct(block))

        if s.check() == sat:
            return True
        else:
            return False


    def get_tensor_unsat_sample_batch(self, batch_size, k=1):
        with torch.no_grad():
            # Start from random sat boards
            indices = torch.randint(0, len(self.sat_boards_cache), (batch_size, k)).to(device)
            boards = self.sat_boards_cache[indices].clone().float()  # [B, K, 4, 4]

            # Choose violation type per sample: 0=row, 1=col, 2=block
            violation_type = torch.randint(0, 3, (batch_size, k)).to(device)  # [B, K]

            # --- Row violation: board[i][1] = board[i][0] ---
            row_i = torch.randint(0, 4, (batch_size, k)).to(device)  # [B, K]
            
            # --- Col violation: board[1][j] = board[0][j] ---
            col_j = torch.randint(0, 4, (batch_size, k)).to(device)  # [B, K]

            # --- Block violation: board[bi][bj+1] = board[bi][bj] ---
            block_i = (torch.randint(0, 2, (batch_size, k)).to(device)) * 2  # 0 or 2
            block_j = (torch.randint(0, 2, (batch_size, k)).to(device)) * 2  # 0 or 2

            # Apply all three violations, then mask to keep only the chosen one
            boards_row = boards.clone()
            boards_col = boards.clone()
            boards_block = boards.clone()

            # Vectorized row violation
            b_idx = torch.arange(batch_size).unsqueeze(1).expand(batch_size, k).to(device)  # [B, K]
            k_idx = torch.arange(k).unsqueeze(0).expand(batch_size, k).to(device)           # [B, K]

            src_val_row = boards[b_idx, k_idx, row_i, 0]           # board[i][0]
            boards_row[b_idx, k_idx, row_i, 1] = src_val_row       # board[i][1] = board[i][0]

            # Vectorized col violation
            src_val_col = boards[b_idx, k_idx, 0, col_j]           # board[0][j]
            boards_col[b_idx, k_idx, 1, col_j] = src_val_col       # board[1][j] = board[0][j]

            # Vectorized block violation
            src_val_block = boards[b_idx, k_idx, block_i, block_j]         # board[bi][bj]
            boards_block[b_idx, k_idx, block_i, block_j + 1] = src_val_block  # board[bi][bj+1] = board[bi][bj]

            # Select which violation applies per (B, K) entry
            is_row   = (violation_type == 0).unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
            is_col   = (violation_type == 1).unsqueeze(-1).unsqueeze(-1)
            is_block = (violation_type == 2).unsqueeze(-1).unsqueeze(-1)

            result = (
                torch.where(is_row,   boards_row,   boards) 
            )
            result = torch.where(is_col,   boards_col,   result)
            result = torch.where(is_block, boards_block, result)

            return result.long()  # [B, K, 4, 4]
        
    def tensor_unsat_sample_batch(self, batch_size, k=1):
        unsat_sample = self.get_tensor_unsat_sample_batch(batch_size, k)  # [B, K, 4, 4]
        
        check = self.tensor_check(unsat_sample.view(-1, 4, 4))  # [B*K]
        while check.any():
            unsat_sample = self.get_tensor_unsat_sample_batch(batch_size, k)  # [B, K, 4, 4]
            check = self.tensor_check(unsat_sample.view(-1, 4, 4))  # [B*K]
        
        return unsat_sample


    def tensor_sat_sample_batch(self, batch_size, k=1):
        with torch.no_grad():
            indices = torch.randint(0, len(self.sat_boards_cache), (batch_size, k)).to(device)
            sat_sample = self.sat_boards_cache[indices]  # Shape: [batch_size, k, 4, 4]
            return sat_sample
        
    def tensor_check(self, candidate_batch):
        with torch.no_grad():
            flat_candidate = candidate_batch.view(candidate_batch.size(0), -1)  # Shape: [batch_size, 16]
            flat_cache = self.sat_boards_cache.view(self.sat_boards_cache.size(0), -1)  # Shape: [num_solutions, 16]
            matches = (flat_candidate.unsqueeze(1) == flat_cache.unsqueeze(0)).all(dim=2)  # Shape: [batch_size, num_solutions]
            return matches.any(dim=1)  # Shape: [batch_size]
    
if __name__ == '__main__':
    sampler = Sampler()
    sat_boards = sampler.tensor_sat_sample_batch(batch_size=5, k=1)
    unsat_boards = sampler.tensor_unsat_sample_batch(batch_size=5, k=1)
    print(unsat_boards)