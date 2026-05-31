import torch
import torchvision
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_DIGITS = 9   # Sudoku 9x9 uses digits 1..9.


def tensorized_load_mnist_by_digit():
    mnist_train = torchvision.datasets.MNIST(
        "datasets/MNIST/train",
        train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    mnist_test = torchvision.datasets.MNIST(
        "datasets/MNIST/test",
        train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    mnist_train_data = mnist_train.data.to(device).to(torch.float32) / 255.0
    mnist_test_data = mnist_test.data.to(device).to(torch.float32) / 255.0

    train_idx = [torch.where(mnist_train.targets == d)[0].to(device) for d in range(1, NUM_DIGITS + 1)]
    test_idx = [torch.where(mnist_test.targets == d)[0].to(device) for d in range(1, NUM_DIGITS + 1)]

    min_train = min(len(t) for t in train_idx)
    min_test = min(len(t) for t in test_idx)

    by_digit_train = torch.stack(
        [mnist_train_data[t][:min_train] for t in train_idx]
    ).unsqueeze(2).to(device)  # [9, L, 1, 28, 28]

    by_digit_test = torch.stack(
        [mnist_test_data[t][:min_test] for t in test_idx]
    ).unsqueeze(2).to(device)  # [9, L, 1, 28, 28]

    return by_digit_train, by_digit_test


def tensorized_get_mnist_digit_board(board, by_digit):
    B, N, _ = board.shape  # [B, 9, 9]
    L = by_digit.shape[1]
    imgs = torch.zeros((B, NUM_DIGITS, NUM_DIGITS, 1, 28, 28), dtype=torch.float32).to(device)
    for i in range(NUM_DIGITS):
        for j in range(NUM_DIGITS):
            indices = torch.randint(0, L, (B,)).to(device)
            digit = board[:, i, j] - 1
            imgs[:, i, j] = by_digit[digit, indices]
    return imgs


def tensorized_get_anchor_mnist_digit(by_digit):
    L = by_digit.shape[1]
    indices = torch.randint(0, L, (NUM_DIGITS,)).to(device)
    anchor_imgs = by_digit[torch.arange(NUM_DIGITS).to(device), indices]  # [9, 1, 28, 28]
    return anchor_imgs


def load_mnist_by_digit():
    mnist_train = torchvision.datasets.MNIST(
        "datasets/MNIST/train",
        train=True, download=True,
        transform=torchvision.transforms.ToTensor()
    )
    mnist_test = torchvision.datasets.MNIST(
        "datasets/MNIST/test",
        train=False, download=True,
        transform=torchvision.transforms.ToTensor()
    )

    by_digit_train = {d: [] for d in range(1, NUM_DIGITS + 1)}
    for img, label in mnist_train:
        if 1 <= label <= NUM_DIGITS:
            by_digit_train[label].append(img)

    by_digit_test = {d: [] for d in range(1, NUM_DIGITS + 1)}
    for img, label in mnist_test:
        if 1 <= label <= NUM_DIGITS:
            by_digit_test[label].append(img)

    return by_digit_train, by_digit_test


def get_mnist_digit_board(board, by_digit):
    rng = np.random.default_rng()
    imgs = []
    for i in range(NUM_DIGITS):
        row = []
        for j in range(NUM_DIGITS):
            img = rng.choice(by_digit[board[i][j]])
            row.append(torch.tensor(img))
        imgs.append(torch.stack(row))
    return torch.stack(imgs)


def get_anchor_mnist_digit(by_digit):
    rng = np.random.default_rng()
    imgs = []
    for key in by_digit:
        idx = rng.integers(len(by_digit[key]))
        img = by_digit[key][idx]
        imgs.append(torch.as_tensor(img))
        if isinstance(by_digit[key], list):
            by_digit[key].pop(idx)
        else:
            by_digit[key] = np.delete(by_digit[key], idx, axis=0)
    return torch.stack(imgs)


if __name__ == '__main__':
    by_digit_train, by_digit_test = tensorized_load_mnist_by_digit()
    print('by_digit_train:', by_digit_train.shape)
    sample_board = torch.arange(1, NUM_DIGITS + 1).repeat(NUM_DIGITS, 1).unsqueeze(0).to(device)
    imgs = tensorized_get_mnist_digit_board(sample_board, by_digit_train)
    print('imgs:', imgs.shape)
    anchor_imgs = tensorized_get_anchor_mnist_digit(by_digit_train)
    print('anchor:', anchor_imgs.shape)
