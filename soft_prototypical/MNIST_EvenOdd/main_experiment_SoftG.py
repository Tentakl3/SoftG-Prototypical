import random
import torch
import pickle
import numpy as np
import os # Added import for os module

from databuilder.MNIST_EvenOdd.MNIST_EvenOdd_full_loader import get_mnist_evenodd_dataset
from .mnistevenodd_SoftG_main import LTN_SoftG_MNISTEvenOdd

if __name__ == '__main__':
    results_tries = {}
    schedules = ["exp", "linear", "log"]
    #schedules = ["exp"]
    for schedule in schedules:
        for i in range(1):
            seed = i*128
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            train_loader, test_loader, anchor_imgs = get_mnist_evenodd_dataset(64)

            ltn_proto_classifier = LTN_SoftG_MNISTEvenOdd(num_classes=10)
            results = ltn_proto_classifier.train(train_loader, test_loader, epochs=10, schedule=schedule)

            results_tries[i] = results

        file_path = file_path = f'results/MNIST_EvenOdd/SoftG/{schedule}-10-SoftG-projection-10epochs-sampling3.pkl'

        # Open the file in write-binary mode and save the dictionary
        with open(file_path, 'wb') as file_handle:
            pickle.dump(results_tries, file_handle)