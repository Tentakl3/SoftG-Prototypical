from z3 import *
import torch
import numpy as np
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pool of pre-computed SAT 9x9 boards. Full SAT space is ~6.7e21 boards; N
# is the working pool used for the train/test split. Boards are derived
# from a single z3 solution plus random symmetry transformations (digit
# relabel, in-band row swap, in-stack column swap, band/stack permutation,
# transpose); these preserve SAT and give sufficient diversity.
N_SAT_CACHE = 2000


class Sampler:
    def __init__(self, n_sat_cache=N_SAT_CACHE):
        self.n_sat_cache = n_sat_cache
        base = self._solve_one()
        boards = self._derive_pool(base, n_sat_cache)
        self.sat_boards_cache = torch.tensor(boards, dtype=torch.long).to(device)
        self.sat_boards_cache_array = self.sat_boards_cache.cpu().tolist()

    def _solve_one(self):
        s = Solver()
        X = [[Int(f"x_{i}_{j}") for j in range(9)] for i in range(9)]
        for i in range(9):
            for j in range(9):
                s.add(And(X[i][j] >= 1, X[i][j] <= 9))
        for i in range(9):
            s.add(Distinct(X[i]))
            s.add(Distinct([X[j][i] for j in range(9)]))
        for i in [0, 3, 6]:
            for j in [0, 3, 6]:
                s.add(Distinct([X[i + di][j + dj] for di in range(3) for dj in range(3)]))
        assert s.check() == sat
        m = s.model()
        return np.array(
            [[m.evaluate(X[i][j]).as_long() for j in range(9)] for i in range(9)]
        )

    @staticmethod
    def _digit_relabel(board, perm):
        # perm: list of length 9 with target labels (1-indexed)
        out = np.zeros_like(board)
        for d in range(1, 10):
            out[board == d] = perm[d - 1]
        return out

    @staticmethod
    def _row_perm(board, band, perm):
        # perm: list of length 3
        idx = [band + p for p in perm]
        out = board.copy()
        out[[band, band + 1, band + 2]] = board[idx]
        return out

    @staticmethod
    def _col_perm(board, stack, perm):
        idx = [stack + p for p in perm]
        out = board.copy()
        out[:, [stack, stack + 1, stack + 2]] = board[:, idx]
        return out

    @staticmethod
    def _band_perm(board, perm):
        # perm: list of length 3 mapping band indices
        bands = [board[3 * p:3 * p + 3] for p in perm]
        return np.vstack(bands)

    @staticmethod
    def _stack_perm(board, perm):
        stacks = [board[:, 3 * p:3 * p + 3] for p in perm]
        return np.hstack(stacks)

    def _random_transform(self, board):
        # Apply a random composition of symmetry-preserving transforms.
        out = board.copy()
        digit_perm = list(range(1, 10))
        random.shuffle(digit_perm)
        out = self._digit_relabel(out, digit_perm)
        for band in [0, 3, 6]:
            perm = list(range(3))
            random.shuffle(perm)
            out = self._row_perm(out, band, perm)
        for stack in [0, 3, 6]:
            perm = list(range(3))
            random.shuffle(perm)
            out = self._col_perm(out, stack, perm)
        band_perm = list(range(3))
        random.shuffle(band_perm)
        out = self._band_perm(out, band_perm)
        stack_perm = list(range(3))
        random.shuffle(stack_perm)
        out = self._stack_perm(out, stack_perm)
        if random.random() < 0.5:
            out = out.T
        return out

    def _derive_pool(self, base, n):
        boards = []
        seen = set()
        attempts = 0
        max_attempts = n * 8
        while len(boards) < n and attempts < max_attempts:
            attempts += 1
            cand = self._random_transform(base)
            key = cand.tobytes()
            if key in seen:
                continue
            seen.add(key)
            boards.append(cand)
        return np.array(boards)

    def get_tensor_unsat_sample_batch(self, batch_size, k=1):
        with torch.no_grad():
            n_sat = self.sat_boards_cache.shape[0]
            indices = torch.randint(0, n_sat, (batch_size, k)).to(device)
            boards = self.sat_boards_cache[indices].clone().float()
            violation_type = torch.randint(0, 3, (batch_size, k)).to(device)
            row_i = torch.randint(0, 9, (batch_size, k)).to(device)
            row_j = torch.randint(0, 8, (batch_size, k)).to(device)
            col_i = torch.randint(0, 8, (batch_size, k)).to(device)
            col_j = torch.randint(0, 9, (batch_size, k)).to(device)
            block_i = (torch.randint(0, 3, (batch_size, k)).to(device)) * 3
            block_j = (torch.randint(0, 3, (batch_size, k)).to(device)) * 3
            block_di = torch.randint(0, 3, (batch_size, k)).to(device)
            block_dj = torch.randint(0, 2, (batch_size, k)).to(device)
            boards_row = boards.clone()
            boards_col = boards.clone()
            boards_block = boards.clone()
            b_idx = torch.arange(batch_size).unsqueeze(1).expand(batch_size, k).to(device)
            k_idx = torch.arange(k).unsqueeze(0).expand(batch_size, k).to(device)
            src_row = boards[b_idx, k_idx, row_i, row_j]
            boards_row[b_idx, k_idx, row_i, row_j + 1] = src_row
            src_col = boards[b_idx, k_idx, col_i, col_j]
            boards_col[b_idx, k_idx, col_i + 1, col_j] = src_col
            target_i = block_i + block_di
            target_j_src = block_j + block_dj
            src_block = boards[b_idx, k_idx, target_i, target_j_src]
            boards_block[b_idx, k_idx, target_i, target_j_src + 1] = src_block
            is_row = (violation_type == 0).unsqueeze(-1).unsqueeze(-1)
            is_col = (violation_type == 1).unsqueeze(-1).unsqueeze(-1)
            is_block = (violation_type == 2).unsqueeze(-1).unsqueeze(-1)
            result = torch.where(is_row, boards_row, boards)
            result = torch.where(is_col, boards_col, result)
            result = torch.where(is_block, boards_block, result)
            return result.long()

    def tensor_unsat_sample_batch(self, batch_size, k=1):
        return self.get_tensor_unsat_sample_batch(batch_size, k)

    def tensor_sat_sample_batch(self, batch_size, k=1):
        with torch.no_grad():
            n_sat = self.sat_boards_cache.shape[0]
            indices = torch.randint(0, n_sat, (batch_size, k)).to(device)
            return self.sat_boards_cache[indices]

    def tensor_check(self, candidate_batch):
        """Membership against SAT cache. Used as a sufficient condition only —
        a non-match does not imply UNSAT for 9x9 (cache covers a small
        fraction of the SAT space)."""
        with torch.no_grad():
            flat_candidate = candidate_batch.view(candidate_batch.size(0), -1)
            flat_cache = self.sat_boards_cache.view(
                self.sat_boards_cache.size(0), -1)
            matches = (flat_candidate.unsqueeze(1) == flat_cache.unsqueeze(0)).all(dim=2)
            return matches.any(dim=1)


if __name__ == '__main__':
    import time
    t0 = time.time()
    sampler = Sampler(n_sat_cache=200)
    print(f'cache built in {time.time()-t0:.1f}s')
    print('cache shape:', sampler.sat_boards_cache.shape)
    print('sat[0]:')
    print(sampler.sat_boards_cache[0].cpu().numpy())
    print('sat[1]:')
    print(sampler.sat_boards_cache[1].cpu().numpy())
    unsat = sampler.tensor_unsat_sample_batch(batch_size=2, k=1)
    print('unsat[0]:')
    print(unsat[0, 0].cpu().numpy())
