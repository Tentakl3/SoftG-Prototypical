from z3 import *
import random
import numpy as np


def sudoku_4x4_solver(switch=False):
    if switch:
        s = Optimize()
    else:
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

def check_sudoku_4x4(board):
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

def generate_unsat_board(board):
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

def generate_sat_boards_set():
    s, X = sudoku_4x4_solver()
    solutions = []

    while s.check() == sat:
        m = s.model()

        board = np.array(
            [[m.evaluate(X[i][j]).as_long() for j in range(4)] for i in range(4)]
        )
        solutions.append([board, 1])

        block = Or([
            X[i][j] != m.evaluate(X[i][j])
            for i in range(4)
            for j in range(4)
        ])
        s.add(block)

    return solutions

def generate_unsat_boards_set(sat_boards):
    neg_solutions = []
    for board, _ in sat_boards:
        correct_board = board.copy()
        corrupted_board = generate_unsat_board(correct_board)
        check = check_sudoku_4x4(board)
        while check:
            correct_board = board.copy()
            corrupted_board = generate_unsat_board(correct_board)
            check = check_sudoku_4x4(corrupted_board)
        neg_solutions.append([corrupted_board, 0])

    return neg_solutions

def generate_boards_set():
    sat_boards = generate_sat_boards_set()
    unsat_boards = generate_unsat_boards_set(sat_boards)

    return sat_boards, unsat_boards