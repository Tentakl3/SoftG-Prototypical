import ltn
import torch
import math
import torch.nn.functional as F
from sklearn.metrics import f1_score

from backbones.Sudoku9x9.CNN_Sudoku import MNISTConv
from samplers.Sudoku9x9.sudoku9x9_sampler import Sampler
from projections.Sudoku9x9.projection_sudoku9x9 import Projection
from ltn_utils.Sudoku9x9.ltn_utils_sudoku9x9 import Logic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Soft_Sudoku:
    def __init__(self, num_classes, layer_sizes=(512, 256, 100, 10), anchor_digits=None):
        self.cnn_s_d = MNISTConv(linear_layers_sizes=(256, 100, 84, num_classes)).to(ltn.device)
        self.num_classes = num_classes
        self.sampler = Sampler()
        self.logical = Logic()
        self.projection = Projection()
        # NOTE(corr-28): one labelled image per class, used by the
        # supervised anchor cross-entropy term in `train`.
        self.anchor_digits = anchor_digits
        self.boards_cache = {}
        self.alpha = 0.05

    def train(self, train_loader, test_loader, epochs, schedule, projection, criteria):
        optimizer = torch.optim.Adam(self.cnn_s_d.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        sampling_epoch = 3

        t = 1
        T = T0 = 1
        K = 144

        train_digit_accs = []
        test_digit_accs = []
        train_board_accs = []
        test_board_accs = []
        train_digit_f1s = []
        test_digit_f1s = []
        train_board_f1s = []
        test_board_f1s = []
        train_times = []

        board_candidate_cache = torch.zeros((len(train_loader.dataset), K, 9, 9), dtype=torch.long).to(device) #[dataset_size, K, 9, 9]
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
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
            self.cnn_s_d.train()
            for batch_idx, (board_images, board_labels, digit_labels, sample_idx) in enumerate(train_loader):
                optimizer.zero_grad()
                board_images = board_images.to(ltn.device) #[64, 4, 4, 1, 28, 28]
                board_labels = board_labels.to(ltn.device) #[64]
                digit_labels = digit_labels.to(ltn.device) #[64, 4, 4]
                sample_idx = sample_idx.to(ltn.device) #[64]

                B, N, _ = digit_labels.shape
                board_images_reshape = board_images.reshape(B*N*N, 1, 28, 28)
                p_digits, _ = self.cnn_s_d(board_images_reshape) #[B*N*N, num_classes]
                p_digits_reshape = p_digits.reshape(B, N*N, self.num_classes) #[B, 16, num_classes]

                logic_labels = (board_labels == 1) #[B]
                idx_sat = sample_idx[logic_labels]
                idx_unsat = sample_idx[~logic_labels]

                if epoch == 0:
                    board_candidate_cache[idx_sat] = self.sampler.tensor_sat_sample_batch(len(idx_sat), K)  #[B, K, 9, 9]
                    board_candidate_cache[idx_unsat] = self.sampler.tensor_unsat_sample_batch(len(idx_unsat), K)  #[B, K, 9, 9]
                elif epoch > sampling_epoch:
                    # For later epochs, we can use the model's current predictions to generate new candidates
                    board_candidate_cache[idx_sat] = self.switch_k_candidate(p_digits_reshape[logic_labels], board_candidate_cache[idx_sat], T, projection, criteria, sat=True)
                    board_candidate_cache[idx_unsat] = self.switch_k_candidate(p_digits_reshape[~logic_labels], board_candidate_cache[idx_unsat], T, projection, criteria, sat=False)
                batch_candidates = board_candidate_cache[sample_idx]  #[B, K, 9, 9]

                target_digits = (batch_candidates - 1).reshape(B, K, N*N)
                # CrossEntropyLoss expects targets in [0, num_classes-1]. Sudoku digits are 1..4.
                CE_loss, best_candidates = self.k_entropy_loss(p_digits.reshape(B, N*N, self.num_classes), target_digits)

                # NOTE(corr-28): supervised cross-entropy on the labelled
                # anchors prevents the row/column/box permutation shortcut.
                if self.anchor_digits is not None:
                    anchor_imgs = self.anchor_digits.to(ltn.device)
                    anchor_logits, _ = self.cnn_s_d(anchor_imgs)
                    anchor_sup_loss = criterion(
                        anchor_logits,
                        torch.arange(self.num_classes, device=ltn.device),
                    )
                else:
                    anchor_sup_loss = torch.zeros((), device=ltn.device)

                total_loss = CE_loss + anchor_sup_loss

                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()

                pred_digits = torch.argmax(p_digits, dim=-1).reshape(B*N*N)
                true_digits = (digit_labels-1).reshape(B*N*N)

                correct_digits = (pred_digits == true_digits).detach()

                correct_train += correct_digits.sum().item()
                total_train += correct_digits.numel()

                all_pred_digits.extend(pred_digits.cpu().numpy())
                all_true_digits.extend(true_digits.cpu().numpy())

                pred_boards = (pred_digits.reshape(B, N, N) + 1)
                
                logical_preds = self.sampler.tensor_check(pred_boards).to(device)  #[B]

                all_pred_boards.append(logical_preds)
                all_true_boards.append(board_labels)

                train_correct_logic_match += (logical_preds == board_labels).sum().item()
                train_total_boards += B

                correct_train_latent += ((best_candidates-1).reshape(B*N*N) == true_digits).sum().item()

            end_time.record()
            torch.cuda.synchronize()
            train_times.append(start_time.elapsed_time(end_time)/1000)

            all_pred_boards = torch.cat(all_pred_boards).detach().cpu().numpy()
            all_true_boards = torch.cat(all_true_boards).detach().cpu().numpy()
            train_digit_f1 = f1_score(all_true_digits, all_pred_digits, average='macro')
            train_board_f1 = f1_score(all_pred_boards, all_true_boards, average='macro')
            train_board_acc = train_correct_logic_match / train_total_boards

            all_pred_digits = []
            all_true_digits = []
            all_pred_boards = []
            all_true_boards = []

            self.cnn_s_d.eval()
            for batch_idx, (board_images, board_labels, digit_labels, sample_idx) in enumerate(test_loader):
                
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

                    pred_boards = (pred_digits.reshape(B, N, N) + 1)

                    logical_preds = self.sampler.tensor_check(pred_boards).to(device)  #[B]

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

            if epoch % 19 == 0:
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
            'test_board_f1s':test_board_f1s,
            'train_latent_acc':train_latent_acc,
            'train_times':train_times
        }

    def board_energy(self, board_costs, sample):
        _, _, N = board_costs.shape
        total_energy = 0.0
        for i in range(N):
            for j in range(N):
                total_energy += board_costs[i, j, sample[i, j] - 1].item()

        return total_energy

    def compute_energy_batch(self, logits, labels):
        log_p = torch.log_softmax(logits, dim=-1)

        B, K, _, _ = labels.shape
        labels_flat = (labels - 1).view(B, K, -1)  # [B, K, 16]

        log_p_expanded = log_p.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, N, C]

        gathered_logp = torch.gather(
            log_p_expanded,
            -1,
            labels_flat.unsqueeze(-1)
        ).squeeze(-1)  # [B, K, N]
        energy = -gathered_logp.mean(dim=-1)  # [B, K]

        return energy
    
    def switch_k_candidate(self, logits, old_samples, T, projection, criteria,sat=True):
        with torch.no_grad():
            B, K, _, _ = old_samples.shape
            if sat:
                if projection == 'on':
                    new_samples = self.projection.tensorized_mutate_sudoku_9x9(old_samples)  #[B, K, 9, 9]
                elif projection == 'off':
                    new_samples = self.sampler.tensor_sat_sample_batch(B*K, 1).reshape(B, K, 9, 9)  #[B, K, 9, 9]
            else:
                if projection == 'on':
                    new_samples = self.projection.tensorized_mutate_sudoku_9x9(old_samples)  #[B, K, 9, 9]
                elif projection == 'off':
                    new_samples = self.sampler.tensor_unsat_sample_batch(B*K, 1).reshape(B, K, 9, 9)  #[B, K, 9, 9]
            
            old_loss = self.compute_energy_batch(logits, old_samples)
            new_loss = self.compute_energy_batch(logits, new_samples)

            delta = -new_loss + old_loss
            v = torch.rand(B, K).to(device)

            try:
                tau = torch.exp(torch.clamp(delta / T, max=0))
            except:
                tau = torch.ones_like(v)

            # NOTE(corr-16): trailing assignment of `accept_mask` after the
            # if/elif disabled the greedy branch; removed.
            if criteria == 'greedy':
                accept_mask = (new_loss < old_loss)
            elif criteria == 'mcmc':
                accept_mask = (new_loss < old_loss) | (v < tau)
            else:
                accept_mask = (new_loss < old_loss) | (v < tau)
            accept_mask_expanded = accept_mask.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
            final_samples = torch.where(accept_mask_expanded, new_samples, old_samples) #type: ignore .long()  # [B, K, 9, 9]

            return final_samples

    def k_entropy_loss(self, logits, samples):
        """
        Instead of free-energy over all K candidates,
        pick the single lowest-energy grounding and apply standard cross-entropy.
        
        logits:  [B, N*N, num_classes]  (raw log-probs or logits from log_softmax)
        samples: [B, K, N*N]            (candidate digit labels, 0-indexed)
        """
        B, K, L = samples.shape

        log_p = torch.log_softmax(logits, dim=-1)  # [B, L, C]

        # Expand for gather: [B, K, L, C]
        log_p_expanded = log_p.unsqueeze(1).expand(-1, K, -1, -1)

        # Gather log-probs of each candidate digit at each position
        gathered_logp = torch.gather(
            log_p_expanded,
            -1,
            samples.unsqueeze(-1)          # [B, K, L, 1]
        ).squeeze(-1)                       # [B, K, L]

        # Energy = negative mean log-prob over positions
        energy = -gathered_logp.mean(dim=-1)  # [B, K]

        # Pick the lowest-energy (highest likelihood) candidate per sample
        best_idx = energy.argmin(dim=-1)      # [B]

        # Gather the best candidate labels: [B, L]
        best_idx_expanded = best_idx.view(B, 1, 1).expand(B, 1, L)
        best_candidates = torch.gather(samples, 1, best_idx_expanded).squeeze(1)  # [B, L]

        # Standard cross-entropy against the best candidate
        # logits is [B, L, C] → need [B, C, L] for F.cross_entropy
        loss = F.cross_entropy(
            logits.permute(0, 2, 1),   # [B, C, L]
            best_candidates,           # [B, L]  (class indices)
            reduction='mean'
        )

        return loss, best_candidates