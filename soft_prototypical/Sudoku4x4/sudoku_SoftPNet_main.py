import ltn
import torch
import math
import torch.nn.functional as F
from sklearn.metrics import f1_score

from backbones.Sudoku4x4.CNN_Sudoku import MNISTConv
from backbones.Sudoku4x4.PNet_Sudoku import LearnableProtoNet_CNN
from samplers.Sudoku4x4.sudoku4x4_sampler import Sampler
from projections.Sudoku4x4.projection_sudoku4x4 import Projection
from ltn_utils.Sudoku4x4.ltn_utils_sudoku4x4 import Logic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftGPNet_Sudoku:
    def __init__(self, num_classes, layer_sizes=(512, 256, 100, 10), anchor_digits=None):
        self.protonet = LearnableProtoNet_CNN(num_classes=num_classes).to(ltn.device)
        self.num_classes = num_classes
        self.sampler = Sampler()
        self.logical = Logic()
        self.projection = Projection()
        self.anchor_digits = anchor_digits
        self.boards_cache = {}
        self.alpha = 0.8

    def train(self, train_loader, test_loader, epochs, schedule, projection, criteria):
        optimizer = torch.optim.Adam(self.protonet.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        anchor_images = self.anchor_digits.to(device) if self.anchor_digits is not None else None
        sampling_epoch = 5

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

        board_candidate_cache = torch.zeros((len(train_loader.dataset), K, 4, 4), dtype=torch.long).to(device) #[dataset_size, K, 4, 4]
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        # NOTE(corr-25): start_time was never .record()'d before the epoch loop
        # in this trainer (the SoftG trainer already does this), so the later
        # call to start_time.elapsed_time(end_time) raised
        # RuntimeError("Both events must be recorded before calculating elapsed time").
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
            self.protonet.train()
            for batch_idx, (board_images, board_labels, digit_labels, sample_idx) in enumerate(train_loader):
                optimizer.zero_grad()
                board_images = board_images.to(ltn.device) #[64, 4, 4, 1, 28, 28]
                board_labels = board_labels.to(ltn.device) #[64]
                digit_labels = digit_labels.to(ltn.device) #[64, 4, 4]
                sample_idx = sample_idx.to(ltn.device) #[64]

                p_norm = F.normalize(self.protonet.prototypes, dim=-1) #[num_classes, z_dim]

                B, N, _ = digit_labels.shape
                board_images_reshape = board_images.reshape(B*N*N, 1, 28, 28)
                z_digits = self.protonet(board_images_reshape)
                # NOTE(corr-11 + corr-27): anchor forward in eval mode so the small
                # anchor set (4 examples) doesn't drive noisy BN batch-stats.
                # corr-27 needs gradients to flow back through z_anchor for the
                # supervised anchor classification loss below, so the no_grad
                # wrapper that corr-11 originally used is removed — eval mode
                # alone is sufficient to silence the BN jitter.
                self.protonet.eval()
                z_anchor = self.protonet(anchor_images)
                self.protonet.train()

                z_digits_reshape = z_digits.reshape(B, N*N, -1)
                z_digits_reshape = F.normalize(z_digits_reshape, dim=-1)
                z_anchor = F.normalize(z_anchor, dim=-1)

                dist = torch.cdist(z_digits_reshape, p_norm) #[B, 16, 4]
                # NOTE(corr-23): keep `score = -dist` as raw "logits" for the
                # MCMC energy + cross-entropy paths, so log_softmax and
                # F.cross_entropy receive logits (not already-softmaxed
                # probabilities). Passing `p_digits = softmax(-dist)` to
                # F.cross_entropy double-softmaxed the input and gave a much
                # smaller / mis-shaped gradient. `p_digits` is kept for argmax
                # readout below (softmax-invariant).
                score = -dist                          # [B, 16, 4]  raw logits
                p_digits = torch.softmax(score, dim=-1)

                anchor_dist = torch.cdist(z_digits_reshape, z_anchor) #[B, 16, 4]
                anchorp_digits = torch.softmax(-anchor_dist, dim=-1)
                anchorp_digits_squared = anchorp_digits.reshape(B, N, N, -1)

                logic_labels = (board_labels == 1) #[B]
                idx_sat = sample_idx[logic_labels]
                idx_unsat = sample_idx[~logic_labels]

                if epoch == 0:
                    board_candidate_cache[idx_sat] = self.sampler.tensor_sat_sample_batch(len(idx_sat), K)  #[B, K, 4, 4]
                    board_candidate_cache[idx_unsat] = self.sampler.tensor_unsat_sample_batch(len(idx_unsat), K)  #[B, K, 4, 4]
                elif epoch > sampling_epoch:
                    # For later epochs, we can use the model's current predictions to generate new candidates
                    board_candidate_cache[idx_sat] = self.switch_k_candidate(score[logic_labels], board_candidate_cache[idx_sat], T, projection, criteria, sat=True)
                    board_candidate_cache[idx_unsat] = self.switch_k_candidate(score[~logic_labels], board_candidate_cache[idx_unsat], T, projection, criteria, sat=False)
                batch_candidates = board_candidate_cache[sample_idx]  #[B, K, 4, 4]

                target_digits = (batch_candidates - 1).reshape(B, K, N*N)
                CE_loss = self.k_entropy_loss(score, target_digits, T)

                z_digits_flat = z_digits_reshape.reshape(B*N*N, -1) #[B*16]
                best_target_digits = self.get_best_candidate(anchorp_digits, target_digits)  #[B, 4, 4]
                best_target_digits_flat = best_target_digits.reshape(B*N*N)  #[B*16]
                #proto_loss = self.prototype_loss_new(z_digits_flat, best_target_digits_flat + 1)

                proto_loss = self.prototype_loss_new(
                    z_digits_flat,              # [B*16, embed_dim]
                    best_target_digits_flat + 1 # [B*16], values in 1..4, ground truth
                )

                # NOTE(corr-26): without this term the prototypes were only
                # pulled toward MCMC's best-of-K guess, which is a permutation-
                # symmetric solution under the Sudoku constraint (row/col/box
                # distinct holds under any digit relabelling). The model thus
                # learned to predict permuted digits and digit accuracy stayed
                # at chance (~25 % for 4 classes) even though board accuracy
                # rose to 0.94. Anchoring prototype i directly to the embedding
                # of the single labelled anchor image for class i+1 grounds the
                # digit identity. Single anchor per class is consistent with
                # our paper's setting.
                proto_anchor_loss = ((F.normalize(z_anchor, dim=-1) - p_norm) ** 2).mean()

                # NOTE(corr-27): direct supervised classification loss on the
                # anchor images. By construction anchor_images[i] is a labelled
                # example of digit class i+1 (per databuilder convention), so
                # the encoder must produce z_anchor[i] closest to prototype[i].
                # corr-26 alone wasn't enough — CE_loss on MCMC's best-of-K
                # candidate dominates the encoder's gradient and pulls z(x)
                # toward permutation-consistent (but identity-wrong) targets.
                # This term gives the encoder one direct label per class — the
                # minimal supervision needed to break the reasoning shortcut,
                # matching the "one labelled datapoint per concept" setting of
                # Andolfi & Giunchiglia (2025).
                anchor_score = -torch.cdist(F.normalize(z_anchor, dim=-1), p_norm)  # [num_classes, num_classes]
                anchor_sup_loss = F.cross_entropy(
                    anchor_score,
                    torch.arange(self.num_classes, device=device),
                )

                total_loss = ((1- self.alpha) * CE_loss
                              + self.alpha * (proto_loss + proto_anchor_loss)
                              + anchor_sup_loss)

                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()


                pred_digits = torch.argmax(p_digits, dim=-1).reshape(B*N*N)
                true_digits = (digit_labels-1).reshape(B*N*N)

                correct_digits = (pred_digits == true_digits).detach()
                correct_latent = (best_target_digits_flat == true_digits).detach()
                correct_train_latent += correct_latent.sum().item()

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

            end_time.record()
            torch.cuda.synchronize()
            train_times.append(start_time.elapsed_time(end_time)/1000)

            all_pred_boards = torch.cat(all_pred_boards).detach().cpu().numpy()
            all_true_boards = torch.cat(all_true_boards).detach().cpu().numpy()
            train_digit_f1 = f1_score(all_true_digits, all_pred_digits, average='macro')
            train_board_f1 = f1_score(all_pred_boards, all_true_boards, average='macro')
            train_board_acc = train_correct_logic_match / train_total_boards
            train_digit_acc = correct_train / total_train

            all_pred_digits = []
            all_true_digits = []
            all_pred_boards = []
            all_true_boards = []

            self.protonet.eval()
            for batch_idx, (board_images, board_labels, digit_labels, sample_idx) in enumerate(test_loader):
                
                with torch.no_grad():
                    board_images = board_images.to(ltn.device)
                    board_labels = board_labels.to(ltn.device)
                    digit_labels = digit_labels.to(ltn.device)

                    p_norm = F.normalize(self.protonet.prototypes, dim=-1)

                    B, N, _ = digit_labels.shape
                    board_images_reshape = board_images.reshape(B*N*N, 1, 28, 28)
                    z_digits = self.protonet(board_images_reshape)
                    #z_anchor = self.protonet(anchor_images)

                    z_digits_reshape = z_digits.reshape(B, N*N, -1)
                    z_digits_reshape = F.normalize(z_digits_reshape, dim=-1)
                    #z_anchor = F.normalize(z_anchor, dim=-1)

                    dist = torch.cdist(z_digits_reshape, p_norm) #[B, 16, 4]
                    p_digits = torch.softmax(-dist, dim=-1)

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
            'embedding':self.protonet.embedding,
            'prototypes':F.normalize(self.protonet.prototypes, dim=-1).cpu().detach().numpy(),
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
                    new_samples = self.projection.tensorized_mutate_sudoku_4x4(old_samples)  #[B, K, 4, 4]
                elif projection == 'off':
                    new_samples = self.sampler.tensor_sat_sample_batch(B*K, 1).reshape(B, K, 4, 4)  #[B, K, 4, 4]
            else:
                if projection == 'on':
                    new_samples = self.projection.tensorized_mutate_sudoku_4x4(old_samples)  #[B, K, 4, 4]
                elif projection == 'off':
                    new_samples = self.sampler.tensor_unsat_sample_batch(B*K, 1).reshape(B, K, 4, 4)  #[B, K, 4, 4]
            
            old_loss = self.compute_energy_batch(logits, old_samples)
            new_loss = self.compute_energy_batch(logits, new_samples)

            delta = -new_loss + old_loss
            v = torch.rand(B, K).to(device)

            try:
                tau = torch.exp(torch.clamp(delta / T, max=0))
            except:
                tau = torch.ones_like(v)

            # NOTE(corr-16): the unconditional `accept_mask = (new_loss<old_loss) | (v<tau)`
            # that previously followed the if/elif overwrote the greedy branch, so the
            # `criteria='greedy'` ablation actually ran MCMC. Removed the overwrite.
            if criteria == 'greedy':
                accept_mask = (new_loss < old_loss)
            elif criteria == 'mcmc':
                accept_mask = (new_loss < old_loss) | (v < tau)
            else:
                accept_mask = (new_loss < old_loss) | (v < tau)
            accept_mask_expanded = accept_mask.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
            final_samples = torch.where(accept_mask_expanded, new_samples, old_samples) #type: ignore .long()  # [B, K, 4, 4]

            return final_samples

    def k_entropy_loss(self, logits, samples, T):
        """
        Pick the single lowest-energy grounding and apply standard cross-entropy.

        logits:  [B, N*N, num_classes]  (raw scores = -dist, NOT probabilities,
                                          see NOTE(corr-23) in train loop)
        samples: [B, K, N*N]            (candidate digit labels, 0-indexed)
        """
        B, K, L = samples.shape

        # NOTE(corr-23): logits are raw scores (-dist) now, so use log_softmax,
        # not log(probs). The downstream F.cross_entropy(logits, ...) below
        # also expects raw logits, so no further change needed there.
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
            best_candidates,           # [B, L]  (class indices, 0-indexed)
            reduction='mean'
        )

        return loss


    def get_best_candidate(self, anchor_logits, candidates):
        with torch.no_grad():
            B, K, _ = candidates.shape
            anchor_log_p = torch.log(anchor_logits)  # [B, 16, C]

            # Expand for gather
            anchor_log_p_expanded = anchor_log_p.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, 16, C]

            # Gather log probs of candidates
            gathered_logp = torch.gather(
                anchor_log_p_expanded,
                -1,
                candidates.unsqueeze(-1)
            ).squeeze(-1)  # [B, K, 16]

            # Energy per candidate (mean over positions)
            energy = -gathered_logp.mean(dim=-1)  # [B, K]

            best_indices = torch.argmin(energy, dim=1)  # [B]
            best_candidates = candidates[torch.arange(B), best_indices]  # [B, 4, 4]

            return best_candidates
    
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

                c_i = z_support.mean(dim=0)
                c_i = F.normalize(c_i, dim=-1)

                class_loss = torch.mean((z_query - c_i) ** 2)

                proto_loss = torch.mean((c_i - p_norm[i]) ** 2)

                total_loss += class_loss + proto_loss

                z_queries[i] = z_query
                centroids[i] = c_i

        return total_loss / len(centroids)