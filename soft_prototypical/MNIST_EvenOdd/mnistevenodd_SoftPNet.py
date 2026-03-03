import ltn
import torch
import numpy as np
from numpy.random.mtrand import rand
import torch.nn.functional as F
import math
import random # Added import
from z3 import *

from backbones.MNIST_EvenOdd.PNet_MNISTEvenOdd import LearnableProtoNet_CNN

class LTN_SoftProto_MNISTEvenOdd:
    def __init__(self, num_classes, anchor_imgs, layer_sizes=(512, 256, 100, 10)):
        #self.protonet = LearnableProtoNet_CNN_MNIST(num_classes=num_classes, layer_sizes=layer_sizes).to(ltn.device)
        self.protonet = LearnableProtoNet_CNN(num_classes=num_classes).to(ltn.device)
        self.num_classes = num_classes
        self.anchor_imgs = anchor_imgs.to(ltn.device)
        self.addition_candidate_cache = {}
        self.pairs_cache = self.get_pairs_cache()
        self.alpha = 0.2

    def train(self, train_loader, test_loader, epochs, schedule):
        optimizer = torch.optim.Adam(self.protonet.parameters(), lr=0.001)
        #optimizer = torch.optim.SGD(self.protonet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        train_sats = []
        test_sats = []
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
                self.protonet.train()
                optimizer.zero_grad()

                addition_labels, mult_labels = addition_labels.to(ltn.device), mult_labels.to(ltn.device)
                operand1_labels, operand2_labels = operand1_labels.to(ltn.device), operand2_labels.to(ltn.device)
                operand_images = operand_images.to(ltn.device)
                image_x, image_y = operand_images[:, 0], operand_images[:, 1]

                z_x = self.protonet(image_x)
                z_y = self.protonet(image_y)
                p_x = torch.softmax(-torch.cdist(z_x, self.protonet.prototypes), dim=1)
                p_y = torch.softmax(-torch.cdist(z_y, self.protonet.prototypes), dim=1)

                T = max(0.1, 1.0 * math.exp(-epoch / 10))

                z = torch.cat([z_x, z_y], dim=0)
                z_anchor = self.protonet(self.anchor_imgs)

                anchorp_x = torch.softmax(-torch.cdist(z_x, z_anchor), dim=1)
                anchorp_y = torch.softmax(-torch.cdist(z_y, z_anchor), dim=1)

                #proto_loss = self.proto_loss(z_anchor)

                z_digits = torch.cat([z_x, z_y], dim=0)

                if epoch < sampling_epoch: #best results just with the get_latent_sample
                    #latent_digits = self.get_latent(anchorp_x, anchorp_y, addition_labels)
                    latent_digits = self.get_latent_sample(anchorp_x, anchorp_y, addition_labels)
                    for i, (idx) in enumerate(sample_idx):
                        if idx.item() not in self.addition_candidate_cache:
                            self.addition_candidate_cache[idx.item()] = latent_digits[i]
                else:
                    latent_digits = torch.stack([self.addition_candidate_cache[idx.item()] for idx in sample_idx])
                    new_latent_digits = self.switch_latent(anchorp_x, anchorp_y, latent_digits, addition_labels, T)
                    for i, (idx) in enumerate(sample_idx):
                        self.addition_candidate_cache[idx.item()] = new_latent_digits[i]

                latent_digits = latent_digits.to(ltn.device)
                logical_loss = self.proto_truth_mnistaddition_soft(p_x, p_y, latent_digits)
                #logical_loss = self.proto_truth_mnistaddition_soft(p_x, p_y, latent_digits)

                z_digits = torch.cat([z_x, z_y], dim=0)
                latent_digits_reshape = torch.cat([latent_digits[:,0], latent_digits[:,1]], dim=0)

                new_proto_loss = self.prototype_loss_new(z_digits, latent_digits_reshape)


                total_loss = 0.6 * new_proto_loss + logical_loss

                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()

                latent_x = latent_digits[:, 0]
                latent_y = latent_digits[:, 1]

                joint_pred_latent = latent_x * self.num_classes + latent_y
                all_joint_pred_latent.append(joint_pred_latent.detach())

                pred_x = torch.argmax(torch.softmax(-torch.cdist(z_x, self.protonet.prototypes), dim=1), dim=1)
                pred_y = torch.argmax(torch.softmax(-torch.cdist(z_y, self.protonet.prototypes), dim=1), dim=1)
                sum_pred = pred_x + pred_y
                train_acc += torch.count_nonzero(torch.eq(addition_labels, sum_pred)) / sum_pred.shape[0]
                all_joint_true_sum.append(addition_labels.detach())
                all_joint_pred_sum.append(sum_pred.detach())
                train_f1 += self.f1_macro_multiclass(addition_labels, sum_pred, 2 * self.num_classes - 1).item()

                train_opeand1_acc += torch.count_nonzero(torch.eq(operand1_labels, pred_x)) / (pred_x.shape[0])
                train_opeand2_acc += torch.count_nonzero(torch.eq(operand2_labels, pred_y)) / (pred_x.shape[0])

                joint_true = operand1_labels * self.num_classes + operand2_labels
                joint_pred = pred_x * self.num_classes + pred_y
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
            train_operands_f1 = self.f1_macro_multiclass(all_joint_true, all_joint_pred, self.num_classes**2).item()
            train_operands_latent_f1 = self.f1_macro_multiclass(all_joint_true, all_joint_pred_latent, self.num_classes**2).item()

            all_joint_pred_sum = torch.cat(all_joint_pred_sum)
            all_joint_true_sum = torch.cat(all_joint_true_sum)
            train_f1 = self.f1_macro_multiclass(all_joint_true_sum, all_joint_pred_sum, 2 * self.num_classes - 1).item()

            train_loss = train_loss / len(train_loader)

            all_joint_true = []
            all_joint_pred = []
            all_joint_true_sum = []
            all_joint_pred_sum = []

            for batch_idx, (operand_images, addition_labels, mult_labels, operand1_labels, operand2_labels, sample_idx) in enumerate(test_loader):
                self.protonet.eval()
                with torch.no_grad():
                    operand_images = operand_images.to(ltn.device)
                    addition_labels, mult_labels = addition_labels.to(ltn.device), mult_labels.to(ltn.device)
                    operand1_labels, operand2_labels = operand1_labels.to(ltn.device), operand2_labels.to(ltn.device)

                    # ground variables with current batch data
                    image_x, image_y = operand_images[:, 0], operand_images[:, 1]
                    labels_n = addition_labels
                    z_x = self.protonet(image_x)
                    z_y = self.protonet(image_y)

                    pred_x = torch.argmax(torch.softmax(-torch.cdist(z_x, self.protonet.prototypes), dim=1), dim=1)
                    pred_y = torch.argmax(torch.softmax(-torch.cdist(z_y, self.protonet.prototypes), dim=1), dim=1)
                    sum_pred = pred_x + pred_y
                    test_acc += torch.count_nonzero(torch.eq(labels_n, sum_pred)) / sum_pred.shape[0]
                    all_joint_true_sum.append(addition_labels.detach())
                    all_joint_pred_sum.append(sum_pred.detach())

                    test_opeand1_acc += torch.count_nonzero(torch.eq(operand1_labels, pred_x)) / (pred_x.shape[0])
                    test_opeand2_acc += torch.count_nonzero(torch.eq(operand2_labels, pred_y)) / (pred_x.shape[0])

                    joint_true = operand1_labels * self.num_classes + operand2_labels
                    joint_pred = pred_x * self.num_classes + pred_y
                    all_joint_true.append(joint_true.detach())
                    all_joint_pred.append(joint_pred.detach())

            all_joint_true = torch.cat(all_joint_true)
            all_joint_pred = torch.cat(all_joint_pred)
            test_operands_f1 = self.f1_macro_multiclass(all_joint_true, all_joint_pred, self.num_classes**2).item()

            all_joint_pred_sum = torch.cat(all_joint_pred_sum)
            all_joint_true_sum = torch.cat(all_joint_true_sum)
            test_f1 = self.f1_macro_multiclass(all_joint_true_sum, all_joint_pred_sum, 2 * self.num_classes - 1).item()

            train_sat = 0.0 #compute_sat_level(train_loader, logical.proto_truth_cifar10_weighted, self.protonet).item()
            test_sat = 0.0 #compute_sat_level(test_loader, logical.proto_truth_weighted_2, self.protonet, self.protonet.prototypes).item()
            train_acc = train_acc / len(train_loader)
            test_acc = test_acc / len(test_loader)

            test_opeand1_acc = test_opeand1_acc / len(test_loader)
            train_opeand1_acc = train_opeand1_acc / len(train_loader)
            test_opeand2_acc = test_opeand2_acc / len(test_loader)
            train_opeand2_acc = train_opeand2_acc / len(train_loader)

            test_opeands_acc = (test_opeand1_acc + test_opeand2_acc) / 2
            train_opeands_acc = (train_opeand1_acc + train_opeand2_acc) / 2

            train_sats.append(train_sat)
            test_sats.append(test_sat)
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
            'test_operands_f1s':test_operands_f1s,
            'embedding':self.protonet.embedding,
            'prototypes':self.protonet.prototypes.cpu().detach().numpy(),
        }

    def proto_loss(self, z_anchor):

        dist = (z_anchor - self.protonet.prototypes)**2
        dist = torch.sum(dist, dim=-1)
        return dist.mean()

    def get_latent(self, p_x, p_y, addition_labels):
        B, C = p_x.shape
        with torch.no_grad():
          pairs = []
          for i in range(B):
              d_x, d_y = self.solve_addition_assignment(p_x[i], p_y[i], addition_labels[i])
              pairs.append((d_x, d_y))

          pairs = torch.tensor(pairs)

        return pairs

    def get_latent_sample(self, p_x, p_y, addition_labels):
        B, C = p_x.shape
        with torch.no_grad():
          pairs = []
          for i in range(B):
              n = addition_labels[i].item()
              samples = self.pairs_cache[n]
              d_x, d_y = self.solve_addition_assignment_sample(p_x[i], p_y[i], samples)
              pairs.append((d_x, d_y))

          pairs = torch.tensor(pairs)

        return pairs

    def f1_macro_multiclass(self, y_true, y_pred, num_classes):
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

    def solve_addition_assignment_sample(self, prob_x, prob_y, sample):

        costs_x = [-math.log(p.item() + 1e-8) for p in prob_x]
        costs_y = [-math.log(p.item() + 1e-8) for p in prob_y]

        costs = []
        for s in sample:
          d1, d2 = s
          costs.append(costs_x[d1] + costs_y[d2])

        min_idx = np.argmin(costs)
        return sample[min_idx]

    def solve_addition_assignment(self, prob_x, prob_y, target_sum):
        opt = Optimize()

        # Define integer variables for the two digits
        d1 = Int('d1')
        d2 = Int('d2')

        # Constraints: Digits must be in range and sum must be correct
        opt.add(d1 >= 0, d1 < self.num_classes)
        opt.add(d2 >= 0, d2 < self.num_classes)
        opt.add(d1 + d2 == int(target_sum))

        # We want to maximize the probability (or minimize -log probability)
        # Since Z3 handles Reals better for optimization:
        costs_x = [-math.log(p.item() + 1e-8) for p in prob_x]
        costs_y = [-math.log(p.item() + 1e-8) for p in prob_y]

        # Mapping the choice of digit to the cost
        cost_expr = Sum([If(d1 == i, costs_x[i], 0) for i in range(self.num_classes)]) + \
                    Sum([If(d2 == i, costs_y[i], 0) for i in range(self.num_classes)])

        opt.minimize(cost_expr)

        if opt.check() == sat:
            model = opt.model()
            return model[d1].as_long(), model[d2].as_long()
        else:
            return torch.argmax(prob_x).item(), torch.argmax(prob_y).item()

    def proto_truth_mnistaddition_soft(self, p_x, p_y, latent_digits):

        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        And = ltn.Connective(ltn.fuzzy_ops.AndMin())
        Andx = ltn.Connective(ltn.fuzzy_ops.AndProd(stable=True))
        Or = ltn.Connective(ltn.fuzzy_ops.OrMax())

        d_1 = ltn.Variable("d_1", latent_digits[:, 0])
        d_2 = ltn.Variable("d_2", latent_digits[:, 1])
        P_x = ltn.Variable("P_x", p_x)
        P_y = ltn.Variable("P_y", p_y)

        Digit_s_d = ltn.Predicate(
            func=lambda x, y: torch.gather(x, 1, y))

        sat_agg = Forall(
            ltn.diag(P_x, d_1, P_y, d_2),
                And(Digit_s_d(P_x, d_1), Digit_s_d(P_y, d_2)), # Use fuzzy equality
                p=2
            ).value

        return (1.-sat_agg)

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

    def switch_latent(self, p_x, p_y, latent_digits, addition_labels, T):
        B, C = latent_digits.shape
        with torch.no_grad():
          new_latent_digits = []
          for i in range(B):
              n = addition_labels[i].item()
              d_x, d_y = latent_digits[i, 0].item(), latent_digits[i, 1].item()
              new_d_x, new_d_y = self.propose_neighbor(d_x, n) #random.choice(self.pairs_cache[n]) #self.propose_neighbor(d_x, n)
              P = - torch.log(p_x[i, d_x]+ 1e-8) - torch.log(p_y[i, d_y]+ 1e-8)
              P_new = - torch.log(p_x[i, new_d_x] + 1e-8) - torch.log(p_y[i, new_d_y] + 1e-8)

              delta = -P_new + P
              tau = torch.exp(delta / T)
              v = torch.rand(1, device=ltn.device)

              if P_new < P or v < tau:
                  new_latent_digits.append((new_d_x, new_d_y))
              else:
                  new_latent_digits.append((d_x, d_y))

          new_latent_digits = torch.tensor(new_latent_digits)
        return new_latent_digits

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

    def prototype_loss_new(self, z_digits, batch_pairs):

      total_loss = 0.0

      centroids = {}
      z_queries = {}
      p_norm = F.normalize(self.protonet.prototypes, dim=-1)
      z_digits = F.normalize(z_digits, dim=-1)
      for i in range(self.num_classes):
          z_i = z_digits[batch_pairs == i]
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