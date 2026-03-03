import torch
import torchvision
import numpy as np

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

    by_digit_train = {1:[], 2:[], 3:[], 4:[]}
    for img, label in mnist_train:
        if label in [1, 2, 3, 4]:
            by_digit_train[label].append(img)

    by_digit_test = {1:[], 2:[], 3:[], 4:[]}
    for img, label in mnist_test:
        if label in [1, 2, 3, 4]:
            by_digit_test[label].append(img)

    return by_digit_train, by_digit_test

def get_mnist_digit_board(board, by_digit):
    rng = np.random.default_rng()
    imgs = []
    for i in range(4):
        row = []
        for j in range(4):
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