import ltn
import torch
import math
import random
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score

projection = ([0,0,1,1,2,2,3,3],[2,3,2,3,0,1,0,1])
proj_index = {'1':[(0,0), (0,1)], '2': [(0,0), (1,0)], '3': [(1,0), (1,1)], '4':[(0,1), (1,1)], '5': [(0,0),(1,1)], \
         '6':[(2,2), (2,3)], '7': [(2,2), (3,2)], '8': [(3,2), (3,3)], '9':[(2,3), (3,3)], '10': [(2,2),(3,3)]}

from backbones.Sudoku4x4.CNN_Sudoku import MNISTConv
from samplers.Sudoku4x4.sudoku4x4_sampler import Sampler
from projections.Sudoku4x4.projection_sudoku4x4 import Projection
from ltn_utils.Sudoku4x4.ltn_utils_Sudoku4x4 import Logic

class Soft_Sudoku:
    def __init__(self, num_classes, layer_sizes=(512, 256, 100, 10)):
        self.cnn_s_d = MNISTConv(linear_layers_sizes=(256, 100, 84, num_classes)).to(ltn.device)
        self.num_classes = num_classes
        self.sampler = Sampler()
        self.logical = Logic()
        self.projection = Projection()
        self.boards_cache = {}
        self.alpha = 0.05
        #self.projection = ([0,0,1,1,2,2,3,3],[2,3,2,3,0,1,0,1])
        self.proj_index = {'1':[(0,0), (0,1)], '2': [(0,0), (1,0)], '3': [(1,0), (1,1)], '4':[(0,1), (1,1)], '5': [(0,0),(1,1)], \
         '6':[(2,2), (2,3)], '7': [(2,2), (3,2)], '8': [(3,2), (3,3)], '9':[(2,3), (3,3)], '10': [(2,2),(3,3)]}

    def train(self, train_loader, test_loader, epochs, schedule):
        optimizer = torch.optim.Adam(self.cnn_s_d.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        sampling_epoch = 3

        t = 1
        T = T0 = 1

        train_digit_accs = []
        test_digit_accs = []
        train_board_accs = []
        test_board_accs = []
        train_digit_f1s = []
        test_digit_f1s = []
        train_board_f1s = []
        test_board_f1s = []

        for epoch in range(epochs):
            train_loss = 0.0
            total_train, correct_train = 0, 0
            total_test, correct_test = 0, 0
            correct_train_latent = 0.0

            all_pred_digits = []
            all_true_digits = []
            all_pred_boards = []
            all_true_boards = []
            boards_correct = 0
            train_total_boards = 0
            test_total_boards = 0
            train_correct_logic_match = 0
            test_correct_logic_match = 0

            p = min(10, 2 + 2*(epoch // 2))
            for batch_idx, (board_images, board_labels, digit_labels, sample_idx) in enumerate(train_loader):
                self.cnn_s_d.train()
                optimizer.zero_grad()
                board_images = board_images.to(ltn.device) #[64, 4, 4, 1, 28, 28]
                board_labels = board_labels.to(ltn.device) #[64]
                digit_labels = digit_labels.to(ltn.device) #[64, 4, 4]

                B, N, _ = digit_labels.shape

                sat_board_mask = (board_labels == 1)
                unsat_board_mask = (board_labels == 0)

                board_images_reshape = board_images.reshape(B*N*N, 1, 28, 28)
                p_digits, features_digits = self.cnn_s_d(board_images_reshape)
                p_digits_reshape = p_digits.reshape(B, N*N, self.num_classes)
                p_digits_squared = p_digits.reshape(B, N, N, -1)

                if epoch < sampling_epoch:
                  latent_boards = self.get_latent(p_digits_squared, board_labels)

                  for i, idx in enumerate(sample_idx):
                      self.boards_cache[idx.item()] = latent_boards[i]

                elif epoch >= sampling_epoch:
                  batch_boards_list = [self.boards_cache[idx.item()].clone() for idx in sample_idx]
                  batch_boards = torch.stack(batch_boards_list).to(device=ltn.device)
                  latent_boards = self.switch_latent(p_digits_squared, board_labels, batch_boards, T)

                  for i, idx in enumerate(sample_idx):
                      self.boards_cache[idx.item()] = latent_boards[i]
                else:
                  batch_boards_list = [self.boards_cache[idx.item()].clone() for idx in sample_idx]
                  latent_boards = torch.stack(batch_boards_list).to(device=ltn.device)
                
                latent_board_reshape = latent_boards.reshape(B, N*N)
                features_digits = features_digits.reshape(B, N*N, -1)
                solved_features = features_digits[sat_board_mask]

                sat_AtomicSudoku = self.logical.SudokuTruth_atomic(p_digits_reshape, latent_board_reshape, p)
                #sat_features = self.logical.neg_SudokuTruth_features(solved_features, p)

                #CE_loss = criterion(p_digits, (latent_boards-1).reshape(B*N*N))

                total_loss = (1. - sat_AtomicSudoku)

                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()

                pred_digits = torch.argmax(p_digits, dim=1).reshape(B*N*N)
                true_digits = (digit_labels-1).reshape(B*N*N)
                latent_digits = (latent_boards-1).reshape(B*N*N)

                correct_digits = (pred_digits == true_digits).detach()
                correct_latent = (latent_digits == true_digits).detach()

                correct_train += correct_digits.sum().item()
                correct_train_latent += correct_latent.sum().item()
                total_train += correct_digits.numel()

                # --- 1. Digit Metrics Preparation ---
                #pred_digits = torch.argmin(dist, dim=-1).reshape(-1)
                #true_digits = (digit_labels-1).reshape(-1)

                all_pred_digits.extend(pred_digits.cpu().numpy())
                all_true_digits.extend(true_digits.cpu().numpy())

                pred_boards = (pred_digits.reshape(B, N, N) + 1).cpu().numpy()
                logical_preds = []

                for i in range(B):
                    is_valid = self.sampler.check_sudoku_4x4(pred_boards[i].tolist())
                    logical_preds.append(1 if is_valid else 0)
                
                logical_preds = torch.tensor(logical_preds).to(ltn.device)

                all_pred_boards.append(logical_preds)
                all_true_boards.append(board_labels)

                train_correct_logic_match += (logical_preds == board_labels).sum().item()
                train_total_boards += B

            all_pred_boards = torch.cat(all_pred_boards).detach().cpu().numpy()
            all_true_boards = torch.cat(all_true_boards).detach().cpu().numpy()
            train_digit_f1 = f1_score(all_true_digits, all_pred_digits, average='macro')
            train_board_f1 = f1_score(all_pred_boards, all_true_boards, average='macro')
            train_board_acc = train_correct_logic_match / train_total_boards

            all_pred_digits = []
            all_true_digits = []
            all_pred_boards = []
            all_true_boards = []

            for batch_idx, (board_images, board_labels, digit_labels, sample_idx) in enumerate(test_loader):
                self.cnn_s_d.eval()
                with torch.no_grad():
                    board_images = board_images.to(ltn.device)
                    board_labels = board_labels.to(ltn.device)
                    digit_labels = digit_labels.to(ltn.device)

                    B, N, _ = board_images.shape[:3]

                    board_images_reshape = board_images.reshape(B*N*N, 1, 28, 28)

                    p_digits, _ = self.cnn_s_d(board_images_reshape)
                    pred_digits = torch.argmax(p_digits, dim=-1).reshape(B*N*N)
                    true_digits = (digit_labels-1).reshape(B*N*N)

                    correct_digits = (pred_digits == true_digits).detach()

                    correct_test += correct_digits.sum().item()
                    total_test += correct_digits.numel()
                    all_pred_digits.extend(pred_digits.cpu().numpy())
                    all_true_digits.extend(true_digits.cpu().numpy())

                    pred_boards = (pred_digits.reshape(B, N, N) + 1).cpu().numpy()
                    logical_preds = []

                    for i in range(B):
                        is_valid = self.sampler.check_sudoku_4x4(pred_boards[i].tolist())
                        logical_preds.append(1 if is_valid else 0)

                    logical_preds = torch.tensor(logical_preds).to(ltn.device)

                    all_pred_boards.append(logical_preds)
                    all_true_boards.append(board_labels)

                    test_correct_logic_match += (logical_preds == board_labels).sum().item()
                    test_total_boards += B

            all_pred_boards = torch.cat(all_pred_boards).detach().cpu().numpy()
            all_true_boards = torch.cat(all_true_boards).detach().cpu().numpy()
            test_digit_f1 = f1_score(all_true_digits, all_pred_digits, average='macro')
            test_board_f1 = f1_score(all_pred_boards, all_true_boards, average='macro')
            test_board_acc = test_correct_logic_match / test_total_boards

            train_digit_acc = correct_train / total_train
            test_digit_acc = correct_test / total_test
            train_loss = train_loss / len(train_loader)

            train_latent_acc = correct_train_latent / total_train

            train_digit_accs.append(train_digit_acc)
            test_digit_accs.append(test_digit_acc)
            train_board_accs.append(train_board_acc)
            test_board_accs.append(test_board_acc)
            train_digit_f1s.append(train_digit_f1)
            test_digit_f1s.append(test_digit_f1)
            train_board_f1s.append(train_board_f1)
            test_board_f1s.append(test_board_f1)


            if epoch >= sampling_epoch:
                if schedule == "exp":
                    T0 = T0 * 0.95
                elif schedule == "linear":
                    dT = 0.05 * 1.0/math.sqrt(t)
                    T0 = T0 - dT
                elif schedule == "log":
                    T0 = T0 / math.log(1+t)
                t+=1
                T = max(0.01, T0)

            if epoch % 1 == 0:
            #if epoch == 9:
                print(" epoch %d | loss %.4f | Train Board Acc %.4f | Train Board F1 %.4f | Train Digit Acc %.4f | Train Digit F1 %.4f | Test Board Acc %.4f | Test Board F1 %.4f | Test Digit Acc %.4f | Test Digit F1 %.4f | Latent Digit Acc %.4f"
                    %(epoch, train_loss, train_board_acc, train_board_f1, train_digit_acc, train_digit_f1, test_board_acc, test_board_f1, test_digit_acc, test_digit_f1, train_latent_acc))
        return{
            'train_digit_accs':train_digit_accs,
            'test_digit_accs':test_digit_accs,
            'train_board_accs':train_board_accs,
            'test_board_accs':test_board_accs,
            'train_digit_f1s':train_digit_f1s,
            'test_digit_f1s':test_digit_f1s,
            'train_board_f1s':train_board_f1s,
            'test_board_f1s':test_board_f1s
        }

    def switch_latent(self, p_digits, board_labels, batch_boards, T):
        B, _, _, N = p_digits.shape
        with torch.no_grad():
          board_costs = -torch.log(p_digits)
          K = 40

          for i, b_label in enumerate(board_labels):
            #samples = random.sample(self.sampler.sat_boards_cache, k=K)
            old_energy = self.board_energy(board_costs[i], batch_boards[i])
            if b_label.item() == 1:
                rand_samples = random.sample(self.sampler.sat_boards_cache, k=K)
                pro_samples = self.projection.mutate_sudoku_4x4(batch_boards[i].detach())
                samples = pro_samples + rand_samples
                new_board, new_energy = self.solve_board_assignment(p_digits[i], board_labels[i], samples)
            else:
                rand_samples = random.sample(self.sampler.sat_boards_cache, k=K)
                pro_samples = self.projection.mutate_sudoku_4x4(batch_boards[i].detach())
                samples = pro_samples + rand_samples
                new_board, new_energy = self.solve_board_assignment(p_digits[i], board_labels[i], samples)
                new_board = self.sampler.generate_unsat_board(new_board)

            delta = -new_energy + old_energy
            v = random.random()
            try:
                tau = math.exp(delta / T)
            except:
                tau = 1.0
            
            if new_energy < old_energy  or v < tau:
                batch_boards[i] = torch.tensor(new_board)

        return batch_boards

    def get_latent(self, p_digits, board_labels):
        B, _, _, N = p_digits.shape
        with torch.no_grad():
          boards = []
          K = 144
          for i, b_label in enumerate(board_labels):
              if b_label.item() == 1:
                samples = random.sample(self.sampler.sat_boards_cache, k=K)
                board, _ = self.solve_board_assignment(p_digits[i], board_labels[i], samples)
              else:
                samples = random.sample(self.sampler.sat_boards_cache, k=K)
                board, _ = self.solve_board_assignment(p_digits[i], board_labels[i], samples)
                board = self.sampler.generate_unsat_board(board)

              boards.append(board)

          boards = np.array(boards)
          boards = torch.tensor(boards).to(device=ltn.device)

        return boards

    def solve_board_assignment(self, p_digits, board_label, samples):
        _, _, N = p_digits.shape
        board_costs = -torch.log(p_digits)
        costs = []
        for s in samples:
            costs.append(self.board_energy(board_costs, s))

        min_idx = np.argmin(costs)
        return samples[min_idx], costs[min_idx]

    def board_energy(self, board_costs, sample):
        _, _, N = board_costs.shape
        total_energy = 0.0
        for i in range(N):
            for j in range(N):
                total_energy += board_costs[i, j, sample[i, j] - 1].item()

        return total_energy