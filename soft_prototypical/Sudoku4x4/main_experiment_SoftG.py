#from sudoku_PNet_main import LTN_Proto_Sudoku
import torch
import random
import numpy as np
import pickle


from databuilder.Sudoku4x4.sudoku_loader import get_mnist_sudoku4x4_dataset, tensorized_get_mnist_sudoku4x4_dataset
from .sudoku_SoftG_main import Soft_Sudoku

if __name__ == '__main__':
    print(torch.cuda.is_available())
    #50 100 300 500
    
    schedules = ["exp", "linear", "log"]
    projections = ['on', 'off']  # 'random' or 'mcmc'
    criteria = ['greedy', 'mcmc']
    #schedules = ["exp"]
    #projections = ['on']
    #criteria = ['mcmc']
    for n_train in [50, 100, 300, 500]:
        for schedule in schedules:
            for projection in projections:
                for criterion in criteria:
                    results_tries = {}
                    for i in range(10):
                        seed = i*128
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        train_loader, test_loader, anchor_digits = tensorized_get_mnist_sudoku4x4_dataset(n_train, n_test=1000, batch_size=64)

                        soft_sudoku = Soft_Sudoku(num_classes=4)
                        results = soft_sudoku.train(train_loader=train_loader, test_loader=test_loader, epochs=20, schedule=schedule, projection=projection, criteria=criterion)
                        results_tries[i] = results
                        del soft_sudoku, train_loader, test_loader, anchor_digits

                    file_path = f'results/Sudoku4x4/SoftG/{schedule}-{projection}-{criterion}-{n_train}train-SoftG-K144-20epochs-sampling3.pkl'

                    with open(file_path, 'wb') as file_handle:
                        pickle.dump(results_tries, file_handle)
