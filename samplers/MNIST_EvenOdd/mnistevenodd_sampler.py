from z3 import *
import torch
import numpy as np
import random

class Sampler:
    def __init__(self):
        self.pairs_cache = self.get_pairs_cache()
        self.tensorized_pairs_cache = self.tensorized_cache()

    def addition_solver(self, n):
        n_val = int(n)

        s = Solver()
        d1, d2 = Int('d1'), Int('d2')
        s.add(d1 >= 0, d1 <= 9, d2 >= 0, d2 <= 9)
        s.add(d1 + d2 == n_val)

        return s, d1, d2

    def get_pairs_cache(self):
        z3_cache = {}
        for n in range(19):
            s, d1, d2 = self.addition_solver(n)

            solutions = []
            while s.check() == sat:
                m = s.model()
                sol = (m[d1].as_long(), m[d2].as_long())
                solutions.append(sol)
                # Block this solution to find the next
                s.add(Or(d1 != sol[0], d2 != sol[1]))
            z3_cache[int(n)] = solutions
        return z3_cache

    def tensorized_cache(self):
        samples = []

        for n in range(19):
            pairs = torch.tensor(self.pairs_cache[n], dtype=torch.long)

            # column containing n
            key = torch.full((pairs.shape[0], 1), n, dtype=torch.long)

            # concatenate as [n, a, b]
            samples.append(torch.cat([key, pairs], dim=1))

        return torch.cat(samples, dim=0)

    def batch_sample(self, n_tensor):
        device = n_tensor.device
        samples = self.tensorized_pairs_cache.to(device)
        cache_n = samples[:, 0]

        mask = n_tensor[:, None] == cache_n[None, :]
        rand = torch.rand(mask.shape, device=device)
        rand[~mask] = -1.0

        idx = rand.argmax(dim=1)

        # return only pairs (a,b)
        return samples[idx, 1:]
    
if __name__ == "__main__":
    sampler = Sampler()

    n_values = torch.tensor([3, 5, 7])
    samples = sampler.batch_sample(n_values)
    print(samples)