import ltn
import torch
import numpy as np
from numpy.random.mtrand import rand
import torch.nn.functional as F
import math
import random # Added import
from z3 import *

from backbones.MNIST_EvenOdd.PNet_MNISTEvenOdd import LearnableProtoNet_CNN
from samplers.MNIST_EvenOdd.mnistevenodd_sampler import Sampler
from projections.MNIST_EvenOdd.projection_mnistevenodd import Projection
from ltn_utils.MNIST_EvenOdd.ltn_utils_mnistevenodd import Logic
from soft_prototypical.MNIST_EvenOdd.sensitivity import SensitivityAnalyzer

class SoftProto_MNISTEvenOdd:
    def __init__(self, num_classes, anchor_imgs, verbose):
        #self.protonet = LearnableProtoNet_CNN_MNIST(num_classes=num_classes, layer_sizes=layer_sizes).to(ltn.device)
        self.protonet = LearnableProtoNet_CNN(num_classes=num_classes).to(ltn.device)
        self.sampler = Sampler()
        self.projection = Projection()
        self.logical = Logic()
        self.num_classes = num_classes
        self.verbose = verbose
        self.anchor_imgs = anchor_imgs.to(ltn.device)
        self.alpha = 0.2

    def train(self, train_loader, test_loader, epochs, schedule, projection, criteria):
        optimizer = torch.optim.Adam(self.protonet.parameters(), lr=0.001)
        #optimizer = torch.optim.SGD(self.protonet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        addition_candidate_cache = torch.empty((len(train_loader.dataset), 2), dtype=torch.long, device=ltn.device)

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
                sample_idx = sample_idx.to(ltn.device)
                image_x, image_y = operand_images[:, 0], operand_images[:, 1]

                z_x = self.protonet(image_x)
                z_y = self.protonet(image_y)
                p_x = torch.softmax(-torch.cdist(z_x, self.protonet.prototypes), dim=1)
                p_y = torch.softmax(-torch.cdist(z_y, self.protonet.prototypes), dim=1)

                # NOTE(corr-1): removed per-batch override `T = max(0.1, exp(-epoch/10))`.
                # T is now driven solely by the schedule branch at the end of the
                # epoch so the `schedule` argument actually distinguishes runs.

                # NOTE(corr-11): forward anchor images in eval mode to avoid noisy
                # batch-stats normalisation of the (size-10) anchor set. q_theta is
                # consumed only by the MCMC kernel under no_grad, so no gradients
                # need to flow back through z_anchor.
                self.protonet.eval()
                with torch.no_grad():
                    z_anchor = self.protonet(self.anchor_imgs)
                self.protonet.train()

                anchorp_x = torch.softmax(-torch.cdist(z_x, z_anchor), dim=1)
                anchorp_y = torch.softmax(-torch.cdist(z_y, z_anchor), dim=1)

                #proto_loss = self.proto_loss(z_anchor)

                z_digits = torch.cat([z_x, z_y], dim=0)

                if epoch < sampling_epoch: #best results just with the get_latent_sample
                    #latent_digits = self.get_latent(anchorp_x, anchorp_y, addition_labels)
                    latent_digits = self.sampler.batch_sample(addition_labels)
                    addition_candidate_cache[sample_idx] = latent_digits
                elif epoch >= sampling_epoch:
                    latent_digits = addition_candidate_cache[sample_idx]
                    new_latent_digits = self.batch_candidate_switch(anchorp_x, anchorp_y, latent_digits, addition_labels, T, projection, criteria)
                    addition_candidate_cache[sample_idx] = new_latent_digits.to(ltn.device)    
                    latent_digits = new_latent_digits

                latent_digits = latent_digits.to(ltn.device)
                logical_loss = self.logical.addition_proto_logic(p_x, p_y, latent_digits)

                z_digits = torch.cat([z_x, z_y], dim=0)
                latent_digits_reshape = torch.cat([latent_digits[:,0], latent_digits[:,1]], dim=0)

                proto_loss = self.prototype_loss(z_digits, latent_digits_reshape)

                total_loss = 0.6 * proto_loss + logical_loss

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
                train_f1 += self.f1_macro_multiclass(addition_labels, sum_pred).item()

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
            test_operands_f1 = self.f1_macro_multiclass(all_joint_true, all_joint_pred).item()

            all_joint_pred_sum = torch.cat(all_joint_pred_sum)
            all_joint_true_sum = torch.cat(all_joint_true_sum)
            test_f1 = self.f1_macro_multiclass(all_joint_true_sum, all_joint_pred_sum).item()

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
                    
        analyze = True
        if analyze == True:
            analyzer = SensitivityAnalyzer(self.protonet, self.num_classes, verbose=self.verbose, device=ltn.device)
            sensitivity_results = analyzer.run_full_analysis(
                data_loader=test_loader,
                anchor_imgs=self.anchor_imgs,
                n_batches=30,
            )
        else:
            sensitivity_results = None

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
            'sensitivity': sensitivity_results,
        }

    def is_valid_digit_pair(self, d_x, d_y):
        return 0 <= d_x < self.num_classes and 0 <= d_y < self.num_classes


    def f1_macro_multiclass(self, y_true, y_pred):
        classes = torch.unique(y_true)
        C = len(classes)

        # (C, N) boolean masks
        pred_mask = y_pred.unsqueeze(0) == classes.unsqueeze(1)  # (C, N)
        true_mask = y_true.unsqueeze(0) == classes.unsqueeze(1)  # (C, N)

        tp = (pred_mask & true_mask).sum(dim=1).float()   # (C,)
        fp = (pred_mask & ~true_mask).sum(dim=1).float()  # (C,)
        fn = (~pred_mask & true_mask).sum(dim=1).float()  # (C,)

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        return f1.mean()

    def switch_latent(self, p_x, p_y, latent_digits, addition_labels, T, projection):
        B, C = latent_digits.shape
        with torch.no_grad():
            new_latent_digits = []
            for i in range(B):
                n = addition_labels[i].item()
                d_x, d_y = latent_digits[i, 0].item(), latent_digits[i, 1].item()

                if projection == 'off':
                    new_d_x, new_d_y = random.choice(self.sampler.pairs_cache[n])
                elif projection == 'on':
                    new_d_x, new_d_y = self.projection.propose_neighbor(d_x, n) #random.choice(self.pairs_cache[n]) #self.propose_neighbor(d_x, n)
         
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

    def batch_candidate_switch(self, p_1, p_2, batch_candidates, addition_labels, T, projection, criteria):
        with torch.no_grad():
            device = ltn.device
            eps = 1e-8

            # Current candidates
            d1 = batch_candidates[:, 0]
            d2 = batch_candidates[:, 1]

            if projection == 'off':
                new_candidates = self.sampler.batch_sample(addition_labels)
            elif projection == 'on':
                new_candidates = self.projection.batch_propose_neighbor(d1, addition_labels)

            new_d1 = new_candidates[:, 0]
            new_d2 = new_candidates[:, 1]

            p_d1 = torch.gather(p_1, 1, d1.unsqueeze(1))
            p_d2 = torch.gather(p_2, 1, d2.unsqueeze(1))

            new_p_d1 = torch.gather(p_1, 1, new_d1.unsqueeze(1))
            new_p_d2 = torch.gather(p_2, 1, new_d2.unsqueeze(1))

            P = p_d1 * p_d2 + eps
            P_new = new_p_d1 * new_p_d2 + eps
            tau = (P_new / P)**(1/T)
            v = torch.rand(tau.shape, device=device)

            # NOTE(corr-22): mirrors corr-2 fix in the SoftG trainer. The previous
            # `(v > tau) | (P_new < P)` reverted on every worse-likelihood proposal
            # regardless of v — greedy hill-climbing, broken detailed balance.
            # Correct Metropolis revert mask: worse AND random rejects, i.e.
            # (v > tau) AND (P_new < P). corr-2 fixed SoftG; this companion fix
            # applies the same to SoftPNet.
            if criteria == 'greedy':
                mask = P_new < P
            elif criteria == 'mcmc':
                mask = (v > tau) & (P_new < P)

            res = torch.where(mask, batch_candidates, new_candidates)

        return res
    
    def prototype_loss(self, z_digits, batch_pairs):
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

        return total_loss / len(centroids)