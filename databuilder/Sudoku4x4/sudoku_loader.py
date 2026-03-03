import torch
import random
import numpy as np
from .mnist_sudoku_loader import get_mnist_digit_board, get_anchor_mnist_digit, load_mnist_by_digit
from .sudoku_builder import generate_boards_set

def split_boards(boards, n_train, n_test):
    rng = np.random.default_rng()
    rng.shuffle(boards)

    train_boards = []
    test_boards = []

    for i in range(n_train):
       train_board = random.choice(boards)
       train_boards.append(train_board)

    for i in range(n_test):
       test_board = random.choice(boards)
       test_boards.append(test_board)


    return train_boards, test_boards

def build_dataset(sat_set, unsat_set, mnist_source, idx):
    imgs = []
    labels = []
    boards = []

    for board, label in sat_set:
        imgs.append(get_mnist_digit_board(board, mnist_source))
        labels.append(label)
        boards.append(board)

    for board, label in unsat_set:
        imgs.append(get_mnist_digit_board(board, mnist_source))
        labels.append(label)
        boards.append(board)

    X = torch.stack(imgs)
    y = torch.tensor(labels)
    boards = torch.tensor(np.array(boards))

    return torch.utils.data.TensorDataset(X, y, boards, idx)

def get_mnist_sudoku4x4_dataset(n_train=100, n_test=25, batch_size=64):

    by_digit_train, by_digit_test = load_mnist_by_digit()
    sat_boards_set, unsat_boards_set = generate_boards_set()

    sat_train, sat_test = split_boards(sat_boards_set, n_train // 2, n_test // 2)
    unsat_train, unsat_test = split_boards(unsat_boards_set, n_train // 2, n_test // 2)

    anchor_digits = get_anchor_mnist_digit(by_digit_train)

    #train_set = build_dataset(sat_train, unsat_train, by_digit_train)
    #test_set  = build_dataset(sat_test,  unsat_test,  by_digit_test)

    train_idx = torch.arange(n_train)
    test_idx = torch.arange(n_test)

    train_set = build_dataset(sat_train, unsat_train, by_digit_train, train_idx)
    test_set  = build_dataset(sat_test, unsat_test, by_digit_test, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, anchor_digits