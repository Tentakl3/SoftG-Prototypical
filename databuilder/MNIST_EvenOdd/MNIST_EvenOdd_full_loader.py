import torch
import torchvision
import numpy as np
import random

def get_anchors(train_imgs, train_labels):
    anchor_label = []
    anchor_image = []

    B, C, H, W = train_imgs.shape

    mask_img = torch.ones(B, dtype=torch.bool)
    mask_label = torch.ones(B, dtype=torch.bool)

    for i in range(B):
      if train_labels[i].item() not in anchor_label:
          anchor_label.append(train_labels[i])
          anchor_image.append(train_imgs[i])
          mask_img[i] = False
          mask_label[i] = False

    anchor_image = torch.stack(anchor_image)
    anchor_label = torch.stack(anchor_label)

    sort_idx = torch.argsort(anchor_label)

    train_imgs = train_imgs[mask_img]
    train_labels = train_labels[mask_label]

    return anchor_image[sort_idx, :]

def sample_pairs(imgs, labels, even_idx, odd_idx, n_examples):
    imgs_1, imgs_2 = [], []
    lab_1, lab_2 = [], []

    for _ in range(n_examples):
        parity = random.randint(0, 1)
        idx_pool = even_idx if parity == 0 else odd_idx

        i1, i2 = random.sample(idx_pool.tolist(), 2)

        imgs_1.append(imgs[i1])
        imgs_2.append(imgs[i2])
        lab_1.append(labels[i1])
        lab_2.append(labels[i2])

    return (
        torch.stack(imgs_1),
        torch.stack(imgs_2),
        torch.tensor(lab_1),
        torch.tensor(lab_2),
    )

def get_mnist_evenodd_dataset(b_size):
    n_train_examples = 6720
    n_test_examples = 960
    n_operands = 2

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

    train_imgs, train_labels = mnist_train.data, mnist_train.targets
    test_imgs, test_labels = mnist_test.data, mnist_test.targets

    train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0
    train_imgs, test_imgs = torch.unsqueeze(train_imgs, 1), torch.unsqueeze(test_imgs, 1)

    even_train = (train_labels % 2 == 0).nonzero(as_tuple=True)[0]
    odd_train  = (train_labels % 2 == 1).nonzero(as_tuple=True)[0]

    even_test = (test_labels % 2 == 0).nonzero(as_tuple=True)[0]
    odd_test  = (test_labels % 2 == 1).nonzero(as_tuple=True)[0]

    anchor_imgs = get_anchors(train_imgs, train_labels)

    train_img1, train_img2, train_lab1, train_lab2 = sample_pairs(
        train_imgs, train_labels, even_train, odd_train, n_train_examples
    )

    test_img1, test_img2, test_lab1, test_lab2 = sample_pairs(
        test_imgs, test_labels, even_test, odd_test, n_test_examples
    )

    imgs_operand_train = torch.stack([train_img1, train_img2], dim=1)
    imgs_operand_test  = torch.stack([test_img1, test_img2], dim=1)

    label_addition_train = train_lab1 + train_lab2
    label_addition_test  = test_lab1 + test_lab2

    label_mult_train = train_lab1 * train_lab2
    label_mult_test  = test_lab1 * test_lab2

    train_idx = torch.arange(n_train_examples)
    test_idx = torch.arange(n_test_examples)

    train_set = torch.utils.data.TensorDataset(
        imgs_operand_train,
        label_addition_train,
        label_mult_train,
        train_lab1,
        train_lab2,
        train_idx
    )

    test_set = torch.utils.data.TensorDataset(
        imgs_operand_test,
        label_addition_test,
        label_mult_test,
        test_lab1,
        test_lab2,
        test_idx
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=b_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=b_size,
        shuffle=False,
    )

    return train_loader, test_loader, anchor_imgs

def get_mnist_addition_dataset(b_size):
    n_train_examples = 30000
    n_test_examples = 5000
    n_operands = 2

    mnist_train = torchvision.datasets.MNIST(
        "/content/drive/MyDrive/Colab-Notebooks/datasets/MNIST/train",
        train=True, download=True,
        transform=torchvision.transforms.ToTensor()
        )

    mnist_test = torchvision.datasets.MNIST(
        "/content/drive/MyDrive/Colab-Notebooks/datasets/MNIST/test",
        train=False, download=True,
        transform=torchvision.transforms.ToTensor()
        )

    train_imgs, train_labels, test_imgs, test_labels = mnist_train.data, mnist_train.targets, \
                                                       mnist_test.data, mnist_test.targets

    train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

    train_imgs, test_imgs = torch.unsqueeze(train_imgs, 1), torch.unsqueeze(test_imgs, 1)

    anchor_imgs = get_anchors(train_imgs, train_labels)

    imgs_operand_train = [train_imgs[i * n_train_examples:i * n_train_examples + n_train_examples]
                          for i in range(n_operands)]
    labels_operand_train = [train_labels[i * n_train_examples:i * n_train_examples + n_train_examples]
                            for i in range(n_operands)]

    imgs_operand_test = [test_imgs[i * n_test_examples:i * n_test_examples + n_test_examples]
                         for i in range(n_operands)]
    labels_operand_test = [test_labels[i * n_test_examples:i * n_test_examples + n_test_examples]
                           for i in range(n_operands)]

    label_addition_train = labels_operand_train[0] + labels_operand_train[1]
    label_addition_test = labels_operand_test[0] + labels_operand_test[1]

    label_mult_train = labels_operand_train[0] * labels_operand_train[1]
    label_mult_test = labels_operand_test[0] * labels_operand_test[1]

    #train_candidates = []
    #for label in label_addition_train:
    #  train_candidates.append(random.choice(dic[label.item()]))
    #train_candidates = torch.tensor(train_candidates)

    train_idx = torch.arange(n_train_examples)
    test_idx = torch.arange(n_test_examples)

    train_set = [torch.stack(imgs_operand_train, dim=1), label_addition_train, label_mult_train, labels_operand_train[0], labels_operand_train[1], train_idx]
    test_set = [torch.stack(imgs_operand_test, dim=1), label_addition_test, label_mult_test, labels_operand_test[0], labels_operand_test[1], test_idx]

    train_set = torch.utils.data.TensorDataset(*train_set)
    test_set = torch.utils.data.TensorDataset(*test_set)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=b_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=b_size,
        shuffle=False,
    )

    return train_loader, test_loader, anchor_imgs