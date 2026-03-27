from z3 import *
import numpy as np
import random

class Sampler:
    def __init__(self):
        self.pairs_cache = self.get_pairs_cache()

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
