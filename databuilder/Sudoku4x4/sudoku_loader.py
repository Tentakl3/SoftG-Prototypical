import torch
import random
import numpy as np
from samplers.Sudoku4x4.sudoku4x4_sampler import Sampler
from .mnist_sudoku_loader import get_mnist_digit_board, get_anchor_mnist_digit, load_mnist_by_digit, tensorized_load_mnist_by_digit, tensorized_get_mnist_digit_board, tensorized_get_anchor_mnist_digit
from .sudoku_builder import generate_boards_set

sampler = Sampler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def tensorized_split_boards(n_train, n_test):
    # NOTE(corr-6): the original tensorized path called
    # `sampler.tensor_sat_sample_batch(n_train // 2)` and
    # `sampler.tensor_sat_sample_batch(n_test // 2)` separately. Both use
    # `torch.randint` with replacement on the SAT cache, so train and test
    # could (and did) share board configurations — leaking label-relevant
    # information into the test set. Fix: shuffle the SAT cache once,
    # slice disjoint train/test windows. UNSAT side is corruption-derived
    # from independently-drawn SAT boards, so we mirror the same disjoint
    # split there.
    sat_pool = sampler.sat_boards_cache.clone()         # [N_sat, 4, 4]
    n_sat = sat_pool.shape[0]
    half_train = n_train // 2
    half_test = n_test // 2
    # Cap if requested sizes exceed pool capacity.
    if half_train + half_test > n_sat:
        half_test = max(0, n_sat - half_train)
    perm = torch.randperm(n_sat, device=device)
    sat_train_idx = perm[:half_train]
    sat_test_idx = perm[half_train:half_train + half_test]
    sat_train_boards = sat_pool[sat_train_idx]
    sat_test_boards = sat_pool[sat_test_idx]
    sat_train_labels = torch.ones(half_train, dtype=torch.long).to(device)
    sat_test_labels = torch.ones(half_test, dtype=torch.long).to(device)

    # UNSAT side: use the same per-side count via the existing batched
    # corrupted-board generator. These are independent of the SAT train/test
    # split because the corruption operator is stochastic per call.
    unsat_train_boards = sampler.tensor_unsat_sample_batch(half_train).squeeze(1)
    unsat_test_boards = sampler.tensor_unsat_sample_batch(half_test).squeeze(1)
    unsat_train_labels = torch.zeros(half_train, dtype=torch.long).to(device)
    unsat_test_labels = torch.zeros(half_test, dtype=torch.long).to(device)

    train_boards = torch.cat([sat_train_boards, unsat_train_boards], dim=0)
    train_labels = torch.cat([sat_train_labels, unsat_train_labels], dim=0)
    test_boards = torch.cat([sat_test_boards, unsat_test_boards], dim=0)
    test_labels = torch.cat([sat_test_labels, unsat_test_labels], dim=0)

    return train_boards, train_labels, test_boards, test_labels

def tensorized_build_dataset(train_boards, test_boards):
    by_digit_train, by_digit_test = tensorized_load_mnist_by_digit()

    anchor_digits = tensorized_get_anchor_mnist_digit(by_digit_train)

    train_imgs = tensorized_get_mnist_digit_board(train_boards, by_digit_train)
    test_imgs = tensorized_get_mnist_digit_board(test_boards, by_digit_test)

    return train_imgs, test_imgs, anchor_digits

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

def tensorized_get_mnist_sudoku4x4_dataset(n_train=100, n_test=25, batch_size=64):

    train_boards, train_labels, test_boards, test_labels = tensorized_split_boards(n_train, n_test)
    train_imgs, test_imgs, anchor_digits = tensorized_build_dataset(train_boards, test_boards)

    train_idx = torch.arange(n_train)
    test_idx = torch.arange(n_test)

    train_set = torch.utils.data.TensorDataset(train_imgs, train_labels, train_boards, train_idx)
    test_set = torch.utils.data.TensorDataset(test_imgs, test_labels, test_boards, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, anchor_digits

if __name__ == '__main__':
    train_loader, test_loader, anchor_digits = tensorized_get_mnist_sudoku4x4_dataset(n_train=100, n_test=1000, batch_size=64)
    print(train_loader.dataset[0])