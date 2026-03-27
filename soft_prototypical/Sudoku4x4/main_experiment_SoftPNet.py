#from sudoku_PNet_main import LTN_Proto_Sudoku
import torch
import random
import numpy as np
import pickle


from databuilder.Sudoku4x4.sudoku_loader import get_mnist_sudoku4x4_dataset
from ploter.Sudoku4x4.visualize_TSNE_Sudoku import visualize_tsne_sudoku
from .sudoku_SoftPNet_main import Proto_Sudoku

if __name__ == "__main__":
    print(torch.cuda.is_available())
    #50 100 300 500
    train_loader, test_loader, anchor_digits = get_mnist_sudoku4x4_dataset(n_train=300, n_test=1000, batch_size=64)
    #schedules = ["exp", "linear", "log"]
    schedules = ["exp"]
    for schedule in schedules:
        results_tries = {}
        for i in range(1):
            seed = i*128
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            ltn_proto_sudoku = Proto_Sudoku(num_classes=4, anchor_digits=anchor_digits)
            results = ltn_proto_sudoku.train(train_loader, test_loader, epochs=20, schedule=schedule)
            results_tries[i] = results
            del ltn_proto_sudoku
            classes = ['1', '2', '3', '4']

            visualize_tsne_sudoku(results=results, loader=test_loader, num_classes=4, class_names=classes, epoch=10)
        file_path = f'results/Sudoku4x4/{schedule}-300train-10-MyApproach-projection-20epochs-sampling3.pkl'

        with open(file_path, 'wb') as file_handle:
            pickle.dump(results_tries, file_handle)

