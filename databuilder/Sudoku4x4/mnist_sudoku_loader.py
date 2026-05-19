import torch
import torchvision
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    mnist_train_targets = mnist_train.targets.to(device)
    mnist_test_data = mnist_test.data.to(device).to(torch.float32) / 255.0
    mnist_test_targets = mnist_test.targets.to(device)

    train_ones_tensor = torch.where(mnist_train.targets == 1)[0].to(device)
    train_twos_tensor = torch.where(mnist_train.targets == 2)[0].to(device)
    train_threes_tensor = torch.where(mnist_train.targets == 3)[0].to(device)
    train_fours_tensor = torch.where(mnist_train.targets == 4)[0].to(device)

    test_ones_tensor = torch.where(mnist_test.targets == 1)[0].to(device)
    test_twos_tensor = torch.where(mnist_test.targets == 2)[0].to(device)
    test_threes_tensor = torch.where(mnist_test.targets == 3)[0].to(device)
    test_fours_tensor = torch.where(mnist_test.targets == 4)[0].to(device)

    min_train = min(len(train_ones_tensor), len(train_twos_tensor), len(train_threes_tensor), len(train_fours_tensor))
    min_test = min(len(test_ones_tensor), len(test_twos_tensor), len(test_threes_tensor), len(test_fours_tensor))

    by_digit_train = torch.stack([
        mnist_train_data[train_ones_tensor][:min_train],
        mnist_train_data[train_twos_tensor][:min_train],
        mnist_train_data[train_threes_tensor][:min_train],
        mnist_train_data[train_fours_tensor][:min_train]
    ]).unsqueeze(2).to(device) #[4, L, 1, 28, 28]

    by_digit_test = torch.stack([
        mnist_test_data[test_ones_tensor][:min_test],
        mnist_test_data[test_twos_tensor][:min_test],
        mnist_test_data[test_threes_tensor][:min_test],
        mnist_test_data[test_fours_tensor][:min_test]
    ]).unsqueeze(2).to(device) #[4, L, 1, 28, 28]

    return by_digit_train, by_digit_test

def tensorized_get_mnist_digit_board(board, by_digit):
    B, N, _ = board.shape #[B, 4, 4]
    L = by_digit.shape[1]
    imgs = torch.zeros((B, 4, 4, 1, 28, 28), dtype=torch.float32).to(device)
    for i in range(4):
        for j in range(4):
            indices = torch.randint(0, L, (B,)).to(device)  #[B]
            digit = board[:, i, j] - 1  #[B]
            imgs[:, i, j] = by_digit[digit, indices]  #[B, 1, 28, 28]
    
    return imgs

def tensorized_get_anchor_mnist_digit(by_digit):
    L = by_digit.shape[1]
    indices = torch.randint(0, L, (4,)).to(device)  #[4]
    anchor_imgs = by_digit[torch.arange(4).to(device), indices]  #[4, 1, 28, 28]
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

if __name__ == '__main__':
    by_digit_train, by_digit_test = tensorized_load_mnist_by_digit()
    imgs = tensorized_get_mnist_digit_board(torch.tensor([[[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]]]), by_digit_train)
    print(imgs.shape)
    anchor_imgs = tensorized_get_anchor_mnist_digit(by_digit_train)
    print(anchor_imgs.shape)