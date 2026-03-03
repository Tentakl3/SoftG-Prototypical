import torch.nn.functional as F
import math
import random # Added import
from sklearn.metrics import f1_score
import numpy as np
import torch
import ltn


from backbones.Sudoku4x4.PNet_Sudoku import LearnableProtoNet_CNN
from samplers.Sudoku4x4.sudoku4x4_sampler import Sampler
from projections.Sudoku4x4.projection_sudoku4x4 import Projection
from ltn_utils.Sudoku4x4.ltn_utils_Sudoku4x4 import Logic


class Proto_Sudoku:
    def __init__(self, num_classes, anchor_digits):
        #self.protonet = LearnableProtoNet_CNN_MNIST(num_classes=num_classes, layer_sizes=layer_sizes).to(ltn.device)
        self.protonet = LearnableProtoNet_CNN(num_classes=num_classes).to(ltn.device)
        self.sampler = Sampler()
        self.projection = Projection()
        self.boards_cache = {}
        self.anchors = anchor_digits
        self.num_classes = num_classes
        self.alpha = 0.6

    def train(self, train_loader, test_loader, epochs, schedule):
        logical = Logic()
        optimizer = torch.optim.Adam(self.protonet.parameters(), lr=0.001)
        anchor_images = self.anchors.to(ltn.device)
        criteria = torch.nn.CrossEntropyLoss()
        sampling_epoch = 3
        T = T0 = t = 1

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

            p = max(10, 2 + epoch // 2)
            for batch_idx, (board_images, board_labels, digit_labels, sample_idx) in enumerate(train_loader):
                self.protonet.train()
                optimizer.zero_grad()

                board_images = board_images.to(ltn.device)
                board_labels = board_labels.to(ltn.device)
                digit_labels = digit_labels.to(ltn.device)

                B, N, _ = digit_labels.shape

                p_norm = F.normalize(self.protonet.prototypes, dim=-1)

                sat_board_mask = (board_labels == 1)
                unsat_board_mask = (board_labels == 0)

                board_images_reshape = board_images.reshape(B*N*N, 1, 28, 28)

                z_digits = self.protonet(board_images_reshape)
                z_anchor = self.protonet(anchor_images)

                z_digits_reshape = z_digits.reshape(B, N*N, -1)
                z_digits_reshape = F.normalize(z_digits_reshape, dim=-1)
                z_anchor = F.normalize(z_anchor, dim=-1)

                dist = torch.cdist(z_digits_reshape, p_norm) #[B, 16, 4]
                p_digits = torch.softmax(-dist, dim=-1)
                p_digits_squared = p_digits.reshape(B, N, N, -1)

                anchor_dist = torch.cdist(z_digits_reshape, z_anchor) #[B, 16, 4]
                anchorp_digits = torch.softmax(-anchor_dist, dim=-1)
                anchorp_digits_squared = anchorp_digits.reshape(B, N, N, -1)

                if epoch == 0:
                  latent_boards = self.get_latent(anchorp_digits_squared, board_labels)

                  for i, idx in enumerate(sample_idx):
                      self.boards_cache[idx.item()] = latent_boards[i]

                elif epoch >= sampling_epoch:
                  batch_boards_list = [self.boards_cache[idx.item()].clone() for idx in sample_idx]
                  batch_boards = torch.stack(batch_boards_list).to(device=ltn.device)
                  latent_boards = self.switch_latent(anchorp_digits_squared, board_labels, batch_boards, T)

                  for i, idx in enumerate(sample_idx):
                      self.boards_cache[idx.item()] = latent_boards[i]
                else:
                    batch_boards_list = [self.boards_cache[idx.item()].clone() for idx in sample_idx]
                    latent_boards = torch.stack(batch_boards_list).to(device=ltn.device)

                latent_board_reshape = latent_boards.reshape(B, N*N)
                sat_SudokuAtomic = logical.SudokuTruth_atomic(p_digits, latent_board_reshape, p)

                z_digits_flaten = z_digits_reshape.reshape(B*N*N, -1)
                latent_board_flaten = latent_board_reshape.reshape(B*N*N)
                new_proto_loss = self.prototype_loss_new(z_digits_flaten, latent_board_flaten)

                total_loss = self.alpha * new_proto_loss + (1.-sat_SudokuAtomic)

                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()

                pred_digits = torch.argmin(dist, dim=-1).reshape(B*N*N)
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
                self.protonet.eval()
                with torch.no_grad():
                    board_images = board_images.to(ltn.device)
                    board_labels = board_labels.to(ltn.device)
                    digit_labels = digit_labels.to(ltn.device)

                    B, N, _ = digit_labels.shape

                    board_images_reshape = board_images.reshape(B*N*N, 1, 28, 28)
                    z_digits = self.protonet(board_images_reshape)
                    z_digits_reshape = z_digits.reshape(B, N*N, -1)

                    z_digits_reshape = F.normalize(z_digits_reshape, dim=-1)
                    p_norm = F.normalize(self.protonet.prototypes, dim=-1)
                    dist = torch.cdist(z_digits_reshape, p_norm) #[B, 16, 4]

                    pred_digits = torch.argmin(dist, dim=-1).reshape(B*N*N)
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
            'embedding':self.protonet.embedding,
            'prototypes':F.normalize(self.protonet.prototypes, dim=-1).cpu().detach().numpy(),
            'train_digit_accs':train_digit_accs,
            'test_digit_accs':test_digit_accs,
            'train_digit_f1s':train_digit_f1s,
            'test_digit_f1s':test_digit_f1s,
            'train_board_accs':train_board_accs,
            'test_board_accs':test_board_accs,
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
              tau = math.exp(delta / T)

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

    def prototype_loss_new(self, z_digits, batch_boards):

        total_loss = 0.0

        centroids = {}
        z_queries = {}
        p_norm = F.normalize(self.protonet.prototypes, dim=-1)
        z_digits = F.normalize(z_digits, dim=-1)
        for i in range(self.num_classes):
            z_i = z_digits[batch_boards == i+1]
            n_i = z_i.shape[0]
            if n_i > 2:
                idx = torch.randperm(n_i)
                q_idx = idx[: n_i // 2]
                s_idx = idx[n_i // 2 :]

                z_support = z_i[s_idx]
                z_query   = z_i[q_idx]

                c_i = torch.mean(z_support, dim=0)

                class_loss = torch.mean((z_query - c_i) ** 2)

                proto_loss = torch.mean((c_i - p_norm[i]) ** 2)

                total_loss += class_loss + proto_loss

                z_queries[i] = z_query
                centroids[i] = c_i

        eps = 1e-8
        for i, q_i in z_queries.items():
            repel_loss = 0.0
            for j, c_j in centroids.items():
                if i != j:
                    #dist = torch.mean((q_i - p_norm[j]) ** 2)
                    dist = torch.mean((q_i - c_j) ** 2)
                    repel_loss += torch.exp(-dist)

            repel_loss = torch.log(repel_loss + eps)
            total_loss += repel_loss

        return total_loss / len(centroids)