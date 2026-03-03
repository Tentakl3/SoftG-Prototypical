import ltn
import torch
import torch.nn.functional as F
import math
import random # Added import
from z3 import *

from backbones.MNIST_EvenOdd.CNN_MNIST import SingleDigitClassifier, LogitsToProbability

class LTN_SoftG_MNISTEvenOdd:
    def __init__(self, num_classes, layer_sizes=(512, 256, 100, 10)):
        self.num_classes = num_classes
        self.addition_candidate_cache = {}
        self.pairs_cache = self.get_pairs_cache()
        self.alpha = 0.2
        self.cnn_s_d = SingleDigitClassifier()
        self.Digit_s_d = ltn.Predicate(LogitsToProbability(self.cnn_s_d)).to(ltn.device)

    def train(self, train_loader, test_loader, epochs, schedule):
        optimizer = torch.optim.Adam(self.Digit_s_d.parameters(), lr=0.001)
        #optimizer = torch.optim.SGD(self.protonet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        train_accs = []
        test_accs = []
        test_operands_accs = []
        train_operands_accs = []

        train_f1s = []
        test_f1s = []
        test_operands_f1s = []
        train_operands_f1s = []
        sampling_epoch = 2

        T = T0 = 1
        t = 1
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            test_acc = 0.0
            test_opeand1_acc = 0.0
            test_opeand2_acc = 0.0
            train_opeand1_acc = 0.0
            train_opeand2_acc = 0.0

            train_f1 = 0.0
            test_f1 = 0.0
            test_operands_f1 = 0.0
            train_operands_f1 = 0.0

            all_joint_true = []
            all_joint_pred = []
            all_joint_pred_latent = []
            all_joint_true_sum = []
            all_joint_pred_sum = []

            for batch_idx, (operand_images, addition_labels, mult_labels, operand1_labels, operand2_labels, sample_idx) in enumerate(train_loader):
                self.cnn_s_d.train()
                optimizer.zero_grad()

                addition_labels, mult_labels = addition_labels.to(ltn.device), mult_labels.to(ltn.device)
                operand1_labels, operand2_labels = operand1_labels.to(ltn.device), operand2_labels.to(ltn.device)
                operand_images = operand_images.to(ltn.device)
                image_x, image_y = operand_images[:, 0], operand_images[:, 1]

                batch_candidates = []

                for (n, idx) in zip(addition_labels, sample_idx):
                    if idx.item() not in self.addition_candidate_cache:
                        u = random.choice(self.pairs_cache[n.item()])
                        self.addition_candidate_cache[idx.item()] = torch.tensor(u).to(ltn.device)
                    batch_candidates.append(self.addition_candidate_cache[idx.item()])

                batch_candidates = torch.stack(batch_candidates)

                p_1 = torch.softmax(self.cnn_s_d(operand_images[:, 0]), dim=1)
                p_2 = torch.softmax(self.cnn_s_d(operand_images[:, 1]), dim=1)

                if epoch >= sampling_epoch:
                    batch_candidates = self.candidate_switch(p_1, p_2, batch_candidates.clone(), addition_labels, T, epoch)

                sat = self.logic_loss(operand_images, batch_candidates, addition_labels)

                total_loss = (1. - sat)

                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()

                latent_x = batch_candidates[:, 0]
                latent_y = batch_candidates[:, 1]

                joint_pred_latent = latent_x * self.num_classes + latent_y
                all_joint_pred_latent.append(joint_pred_latent.detach())

                predictions_1 = torch.argmax(p_1, dim=1)
                predictions_2 = torch.argmax(p_2, dim=1)
                predictions_sum = predictions_1 + predictions_2

                train_acc += torch.count_nonzero(torch.eq(addition_labels, predictions_sum)) / predictions_sum.shape[0]
                all_joint_true_sum.append(addition_labels.detach())
                all_joint_pred_sum.append(predictions_sum.detach())

                train_opeand1_acc += torch.count_nonzero(torch.eq(operand1_labels, predictions_1)) / (predictions_1.shape[0])
                train_opeand2_acc += torch.count_nonzero(torch.eq(operand2_labels, predictions_2)) / (predictions_2.shape[0])

                joint_true = operand1_labels * self.num_classes + operand2_labels
                joint_pred = predictions_1 * self.num_classes + predictions_2
                all_joint_true.append(joint_true.detach())
                all_joint_pred.append(joint_pred.detach())

            if epoch >= sampling_epoch:
                if schedule == 'linear':
                    dT = 0.05 * 1.0/math.sqrt(t)
                    T0 = T0 - dT
                elif schedule == 'exp':
                    T0 = T0 * 0.95
                elif schedule == 'log':
                    T0 = T0 / math.log(1+t)
                t+=1
                T = max(0.01, T0)

            all_joint_true = torch.cat(all_joint_true)
            all_joint_pred = torch.cat(all_joint_pred)
            all_joint_pred_latent = torch.cat(all_joint_pred_latent)
            train_operands_f1 = self.f1_macro_multiclass(all_joint_true, all_joint_pred).item()
            train_operands_latent_f1 = self.f1_macro_multiclass(all_joint_true, all_joint_pred_latent).item()

            all_joint_pred_sum = torch.cat(all_joint_pred_sum)
            all_joint_true_sum = torch.cat(all_joint_true_sum)
            train_f1 = self.f1_macro_multiclass(all_joint_true_sum, all_joint_pred_sum).item()

            train_loss = train_loss / len(train_loader)

            all_joint_true = []
            all_joint_pred = []
            all_joint_true_sum = []
            all_joint_pred_sum = []

            for batch_idx, (operand_images, addition_labels, mult_labels, operand1_labels, operand2_labels, sample_idx) in enumerate(test_loader):
                self.cnn_s_d.eval()
                with torch.no_grad():
                    addition_labels, mult_labels = addition_labels.to(ltn.device), mult_labels.to(ltn.device)
                    operand1_labels, operand2_labels = operand1_labels.to(ltn.device), operand2_labels.to(ltn.device)
                    operand_images = operand_images.to(ltn.device)
                    image_x, image_y = operand_images[:, 0], operand_images[:, 1]

                    p_1 = torch.softmax(self.cnn_s_d(operand_images[:, 0]), dim=1)
                    p_2 = torch.softmax(self.cnn_s_d(operand_images[:, 1]), dim=1)

                    predictions_1 = torch.argmax(p_1, dim=1)
                    predictions_2 = torch.argmax(p_2, dim=1)
                    predictions_sum = predictions_1 + predictions_2

                    test_acc += torch.count_nonzero(torch.eq(addition_labels, predictions_sum)) / predictions_sum.shape[0]
                    all_joint_true_sum.append(addition_labels.detach())
                    all_joint_pred_sum.append(predictions_sum.detach())

                    test_opeand1_acc += torch.count_nonzero(torch.eq(operand1_labels, predictions_1)) / (predictions_1.shape[0])
                    test_opeand2_acc += torch.count_nonzero(torch.eq(operand2_labels, predictions_2)) / (predictions_2.shape[0])

                    joint_true = operand1_labels * self.num_classes + operand2_labels
                    joint_pred = predictions_1 * self.num_classes + predictions_2
                    all_joint_true.append(joint_true.detach())
                    all_joint_pred.append(joint_pred.detach())

            all_joint_true = torch.cat(all_joint_true)
            all_joint_pred = torch.cat(all_joint_pred)
            test_operands_f1 = self.f1_macro_multiclass(all_joint_true, all_joint_pred).item()

            all_joint_pred_sum = torch.cat(all_joint_pred_sum)
            all_joint_true_sum = torch.cat(all_joint_true_sum)
            test_f1 = self.f1_macro_multiclass(all_joint_true_sum, all_joint_pred_sum).item()

            train_acc = train_acc / len(train_loader)
            test_acc = test_acc / len(test_loader)

            test_opeand1_acc = test_opeand1_acc / len(test_loader)
            train_opeand1_acc = train_opeand1_acc / len(train_loader)
            test_opeand2_acc = test_opeand2_acc / len(test_loader)
            train_opeand2_acc = train_opeand2_acc / len(train_loader)

            test_opeands_acc = (test_opeand1_acc + test_opeand2_acc) / 2
            train_opeands_acc = (train_opeand1_acc + train_opeand2_acc) / 2

            train_accs.append(train_acc)
            test_accs.append(test_acc)
            train_f1s.append(train_f1)
            test_f1s.append(test_f1)
            test_operands_accs.append(test_opeands_acc)
            train_operands_accs.append(train_opeands_acc)
            train_operands_f1s.append(train_operands_f1)
            test_operands_f1s.append(test_operands_f1)

            # we print metrics every 20 epochs of training
            if epoch%1 == 0:
                print(" epoch %d | loss %.4f | Train Acc %.3f | Test Acc %.3f | Train Operands Acc %.3f | Test Operands Acc %.3f | Train Operands F1 %.3f | Test Operands F1 %.3f | Train Latent Operands F1 %.3f"
                    %(epoch, train_loss, train_acc, test_acc, train_opeands_acc, test_opeands_acc, train_operands_f1, test_operands_f1, train_operands_latent_f1))
        return{
            'train_accs':train_accs,
            'test_accs':test_accs,
            'train_f1s':train_f1s,
            'test_f1s':test_f1s,
            'test_operands_accs':test_operands_accs,
            'train_operands_accs':train_operands_accs,
            'train_operands_f1s':train_operands_f1s,
            'test_operands_f1s':test_operands_f1s
        }

    def candidate_switch(self, p_1, p_2, batch_candidates, addition_labels, T, epoch):
        with torch.no_grad():
          device = ltn.device
          eps = 1e-8

          # Current candidates
          d1 = batch_candidates[:, 0]
          d2 = batch_candidates[:, 1]

          # Propose new candidates (same-parity, correct sum)

          if epoch <= 4:
              u = [random.choice(self.pairs_cache[n.item()]) for n in addition_labels]
          else:
              #u = [random.choice(self.pairs_cache[n.item()]) for n in addition_labels]
              u = [self.propose_neighbor(dx, n) for dx, n in zip(d1, addition_labels)]
          new_candidates = torch.tensor(u, device=device)

          new_d1 = new_candidates[:, 0]
          new_d2 = new_candidates[:, 1]

          p_d1 = torch.gather(p_1, 1, d1.unsqueeze(1))
          p_d2 = torch.gather(p_2, 1, d2.unsqueeze(1))

          new_p_d1 = torch.gather(p_1, 1, new_d1.unsqueeze(1))
          new_p_d2 = torch.gather(p_2, 1, new_d2.unsqueeze(1))

          for idx, (p1, p2, n_p1, n_p2) in enumerate(zip(p_d1, p_d2, new_p_d1, new_p_d2)):
            #P = torch.min(p1, p2) + eps
            #P_new = torch.min(n_p1, n_p2) + eps

            P = p1 * p2 + eps
            P_new = n_p1 * n_p2 + eps

            #delta_E = -torch.log(P_new) + torch.log(P)
            #tau = torch.exp(-delta_E / T)

            tau = (P_new / P)**(1/T)
            v = torch.rand(1, device=device)

            if (v > tau) or (P_new < P):
                new_candidates[idx] = batch_candidates[idx]

        return new_candidates

    def logic_loss(self, operand_images, batch_candidates, addition_labels):
        ltn_AndMin = ltn.Connective(ltn.fuzzy_ops.AndMin()) # Renamed to avoid collision with z3.And
        ltn_exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")

        images_x = ltn.Variable("x", operand_images[:, 0])
        images_y = ltn.Variable("y", operand_images[:, 1])
        labels_n = ltn.Variable("n", addition_labels)

        d_1 = ltn.Variable("d_1", batch_candidates[:, 0])
        d_2 = ltn.Variable("d_2", batch_candidates[:, 1])

        #d_1 = ltn.Variable("d_1", torch.tensor(range(10)))
        #d_2 = ltn.Variable("d_2", torch.tensor(range(10)))

        sat_agg = Forall(
            ltn.diag(images_x, d_1, images_y, d_2),
                ltn_AndMin(self.Digit_s_d(images_x, d_1), self.Digit_s_d(images_y, d_2)), # Use fuzzy equality
                p=2
            ).value

        return sat_agg

    def f1_macro_multiclass(self, y_true, y_pred):
        classes = torch.unique(y_true)
        f1_per_class = []

        for c in classes:
            tp = torch.sum((y_pred == c) & (y_true == c)).float()
            fp = torch.sum((y_pred == c) & (y_true != c)).float()
            fn = torch.sum((y_pred != c) & (y_true == c)).float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_per_class.append(f1)

        return torch.mean(torch.stack(f1_per_class))


    def propose_neighbor(self, dx, n):
        n_val = int(n)
        original_dx = int(dx)
        original_dy = n_val - original_dx # Assuming original_dy is also in [0,9]

        possible_deltas = [-1, 1]
        random.shuffle(possible_deltas)

        for delta in possible_deltas:
            proposed_dx = original_dx + delta
            proposed_dy = n_val - proposed_dx

            if 0 <= proposed_dx <= 9 and 0 <= proposed_dy <= 9:
                return [proposed_dx, proposed_dy]

        # If no valid neighbor found with delta -1 or 1, stick to the original valid pair
        return [original_dx, original_dy] # Fallback to original pair, assuming it was valid

    def get_pairs_cache(self):
        z3_cache = {}
        for n in range(19):
            n_val = int(n)

            s = Solver()
            d1, d2 = Int('d1'), Int('d2')
            s.add(d1 >= 0, d1 <= 9, d2 >= 0, d2 <= 9)
            s.add(d1 + d2 == n_val)

            solutions = []
            while s.check() == sat:
                m = s.model()
                sol = (m[d1].as_long(), m[d2].as_long())
                solutions.append(sol)
                # Block this solution to find the next
                s.add(Or(d1 != sol[0], d2 != sol[1]))
            z3_cache[n_val] = solutions
        return z3_cache
