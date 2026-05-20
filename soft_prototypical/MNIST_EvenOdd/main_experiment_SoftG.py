import random
import torch
import pickle
import numpy as np
import os # Added import for os module

from databuilder.MNIST_EvenOdd.MNIST_EvenOdd_full_loader import get_mnist_evenodd_dataset
from .mnistevenodd_SoftG_main import SoftG_MNISTEvenOdd

if __name__ == '__main__':
    results_tries = {}
    schedules = ["exp", "linear", "log"]
    projections = ['on', 'off']  # 'random' or 'mcmc'
    criteria = ['greedy', 'mcmc']
    for schedule in schedules:
        for projection in projections:
            for criterion in criteria:
                for i in range(10):
                    seed = i*128
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    train_loader, test_loader, anchor_imgs = get_mnist_evenodd_dataset(64)

                    softG_classifier = SoftG_MNISTEvenOdd(num_classes=10)
                    results = softG_classifier.train(train_loader, test_loader, epochs=10, schedule=schedule, projection=projection, criteria=criterion)

                    results_tries[i] = results

                file_path = file_path = f'results/MNIST_EvenOdd/SoftG/{schedule}-{projection}-{criterion}-SoftG-10epochs-sampling3.pkl'

        # Open the file in write-binary mode and save the dictionary
        with open(file_path, 'wb') as file_handle:
            pickle.dump(results_tries, file_handle)