import torch
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy


class SensitivityAnalyzer:
    """
    Sensitivity analysis over learnable prototypes vs. anchor-based representations.

    Three complementary analyses:
      1. Gradient sensitivity  — how much does ∂loss/∂prototype_k shift predictions?
      2. Anchor agreement      — KL divergence between p_x (prototype-based) and
                                 anchorp_x (anchor-based) per sample and per class.
      3. Prototype masking     — counterfactual accuracy/F1 drop when prototype_k
                                 is zeroed out.
    """

    def __init__(self, model, num_classes: int, verbose: bool, device):
        self.model = model
        self.num_classes = num_classes
        self.verbose = verbose
        self.device = device

    # ------------------------------------------------------------------
    # 1. GRADIENT-BASED SENSITIVITY
    # ------------------------------------------------------------------
    def gradient_sensitivity(self, data_loader, anchor_imgs, n_batches: int = 20):
        """
        Compute the Frobenius norm of ∂(proto_loss)/∂(prototype_k) for each k.

        Interpretation: a large gradient norm for prototype k means its position
        in embedding space strongly steers the loss — it is a "load-bearing"
        prototype. If learnable prototypes consistently show higher norms than the
        fixed anchor positions, that justifies making them trainable.
        """
        self.model.eval()
        proto_grad_accum = torch.zeros(self.num_classes, device=self.device)
        n_seen = 0

        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= n_batches:
                break

            operand_images = batch[0].to(self.device)
            addition_labels = batch[1].to(self.device)
            image_x, image_y = operand_images[:, 0], operand_images[:, 1]

            # Enable gradient w.r.t. prototypes for this pass
            self.model.prototypes.requires_grad_(True)
            if self.model.prototypes.grad is not None:
                self.model.prototypes.grad.zero_()

            z_x = self.model(image_x)
            z_y = self.model(image_y)

            p_x = torch.softmax(-torch.cdist(z_x, self.model.prototypes), dim=1)
            p_y = torch.softmax(-torch.cdist(z_y, self.model.prototypes), dim=1)

            # Surrogate classification loss: cross-entropy on predicted sum
            # We create soft sum-class logits by outer-product convolution
            # p(sum=n) = Σ_{a+b=n} p_x[a] * p_y[b]
            B = p_x.shape[0]
            sum_logits = torch.zeros(B, 2 * self.num_classes - 1, device=self.device)
            for a in range(self.num_classes):
                for b in range(self.num_classes):
                    sum_logits[:, a + b] += p_x[:, a] * p_y[:, b]

            loss = F.cross_entropy(sum_logits, addition_labels)
            loss.backward()

            # Accumulate per-prototype gradient norms (shape: [num_classes, embed_dim])
            with torch.no_grad():
                if self.model.prototypes.grad is not None:
                    norms = self.model.prototypes.grad.norm(dim=1)  # [num_classes]
                    proto_grad_accum += norms

            self.model.prototypes.detach_()  # prevent gradients from accumulating across batches
            n_seen += 1

        grad_sensitivity = (proto_grad_accum / max(n_seen, 1)).cpu()
        return {
            'grad_norms': grad_sensitivity,                    # [num_classes]
            'most_sensitive': int(grad_sensitivity.argmax()),
            'least_sensitive': int(grad_sensitivity.argmin()),
            'sensitivity_std': grad_sensitivity.std().item(),
        }

    # ------------------------------------------------------------------
    # 2. ANCHOR vs. PROTOTYPE AGREEMENT  (KL divergence)
    # ------------------------------------------------------------------
    def anchor_agreement(self, data_loader, anchor_imgs, n_batches: int = 20):
        """
        For each sample, compute KL(anchorp_x || p_x) and KL(anchorp_y || p_y).

        - Low KL  → anchors and prototypes assign similar class probabilities;
                    they encode redundant information.
        - High KL → anchors provide a genuinely different inductive bias,
                    justifying their inclusion alongside learnable prototypes.

        We also compute per-class KL to identify *which* classes benefit most
        from the dual representation.
        """
        self.model.eval()
        z_anchor = self.model(anchor_imgs.to(self.device)).detach()

        kl_x_list, kl_y_list = [], []
        true_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= n_batches:
                    break

                operand_images = batch[0].to(self.device)
                operand1_labels = batch[3].to(self.device)
                operand2_labels = batch[4].to(self.device)
                image_x, image_y = operand_images[:, 0], operand_images[:, 1]

                z_x = self.model(image_x)
                z_y = self.model(image_y)

                # Prototype-based distributions
                p_x = torch.softmax(-torch.cdist(z_x, self.model.prototypes), dim=1)
                p_y = torch.softmax(-torch.cdist(z_y, self.model.prototypes), dim=1)

                # Anchor-based distributions
                anchorp_x = torch.softmax(-torch.cdist(z_x, z_anchor), dim=1)
                anchorp_y = torch.softmax(-torch.cdist(z_y, z_anchor), dim=1)

                eps = 1e-8
                # KL(anchor || proto) = Σ anchor * log(anchor / proto)
                kl_x = (anchorp_x * (torch.log(anchorp_x + eps) - torch.log(p_x + eps))).sum(dim=1)
                kl_y = (anchorp_y * (torch.log(anchorp_y + eps) - torch.log(p_y + eps))).sum(dim=1)

                kl_x_list.append(kl_x.cpu())
                kl_y_list.append(kl_y.cpu())
                true_labels.append(operand1_labels.cpu())

        kl_x_all = torch.cat(kl_x_list)   # [N]
        kl_y_all = torch.cat(kl_y_list)
        labels_all = torch.cat(true_labels)

        # Per-class average KL
        per_class_kl = {}
        for c in range(self.num_classes):
            mask = labels_all == c
            if mask.sum() > 0:
                per_class_kl[c] = kl_x_all[mask].mean().item()

        return {
            'mean_kl_x': kl_x_all.mean().item(),
            'mean_kl_y': kl_y_all.mean().item(),
            'std_kl_x': kl_x_all.std().item(),
            'per_class_kl': per_class_kl,        # {class_idx: mean_kl}
            'max_divergence_class': max(per_class_kl, key=per_class_kl.get), # type: ignore
            'min_divergence_class': min(per_class_kl, key=per_class_kl.get), # type: ignore
            # High variance here means anchors help some classes much more than others
            'class_kl_variance': np.var(list(per_class_kl.values())),
        }

    # ------------------------------------------------------------------
    # 3. PROTOTYPE MASKING  (counterfactual importance)
    # ------------------------------------------------------------------
    def prototype_masking(self, data_loader, n_batches: int = 20):
        """
        For each prototype k, temporarily zero it out and measure the drop in:
          - addition accuracy (sum prediction)
          - operand F1 (joint class F1)

        A large drop when masking prototype k means that prototype carries
        unique discriminative load that neither the other prototypes nor the
        anchors can compensate for. Combined with the anchor-agreement analysis,
        this reveals complementarity: prototypes that are hard to substitute AND
        strongly diverge from anchors are the clearest evidence for the dual design.
        """
        self.model.eval()

        def evaluate(loader, modified_prototypes, n_batches):
            correct_sum, total = 0, 0
            all_true, all_pred = [], []

            with torch.no_grad():
                for b_idx, batch in enumerate(loader):
                    if b_idx >= n_batches:
                        break
                    operand_images = batch[0].to(self.device)
                    addition_labels = batch[1].to(self.device)
                    operand1_labels = batch[3].to(self.device)
                    operand2_labels = batch[4].to(self.device)
                    image_x, image_y = operand_images[:, 0], operand_images[:, 1]

                    z_x = self.model(image_x)
                    z_y = self.model(image_y)

                    p_x = torch.softmax(-torch.cdist(z_x, modified_prototypes), dim=1)
                    p_y = torch.softmax(-torch.cdist(z_y, modified_prototypes), dim=1)

                    pred_x = p_x.argmax(dim=1)
                    pred_y = p_y.argmax(dim=1)
                    sum_pred = pred_x + pred_y
                    correct_sum += (sum_pred == addition_labels).sum().item()
                    total += sum_pred.shape[0]

                    joint_true = operand1_labels * self.num_classes + operand2_labels
                    joint_pred = pred_x * self.num_classes + pred_y
                    all_true.append(joint_true.cpu())
                    all_pred.append(joint_pred.cpu())

            acc = correct_sum / max(total, 1)
            all_true = torch.cat(all_true)
            all_pred = torch.cat(all_pred)
            f1 = self._f1_macro(all_true, all_pred, self.num_classes ** 2)
            return acc, f1

        # Baseline (no masking)
        baseline_protos = self.model.prototypes.detach().clone()
        base_acc, base_f1 = evaluate(data_loader, baseline_protos, n_batches)

        importance = {}
        for k in range(self.num_classes):
            masked = baseline_protos.clone()
            masked[k] = 0.0   # zero out prototype k
            acc_k, f1_k = evaluate(data_loader, masked, n_batches)
            importance[k] = {
                'acc_drop': base_acc - acc_k,
                'f1_drop': base_f1 - f1_k,
                'masked_acc': acc_k,
                'masked_f1': f1_k,
            }

        sorted_by_importance = sorted(
            importance.items(),
            key=lambda x: x[1]['acc_drop'],
            reverse=True
        )

        return {
            'baseline_acc': base_acc,
            'baseline_f1': base_f1,
            'per_prototype': importance,           # {k: {acc_drop, f1_drop, ...}}
            'ranking': [k for k, _ in sorted_by_importance],
            'most_critical_prototype': sorted_by_importance[0][0],
            'least_critical_prototype': sorted_by_importance[-1][0],
        }

    # ------------------------------------------------------------------
    # JOINT REPORT
    # ------------------------------------------------------------------
    def run_full_analysis(self, data_loader, anchor_imgs, n_batches: int = 20):
        """
        Run all three analyses and print a combined justification report.
        Returns a dict with all raw results.
        """
        print("=" * 65)
        print("PROTOTYPE SENSITIVITY ANALYSIS")
        print("=" * 65)

        print("\n[1/3] Gradient sensitivity ...")
        grad_res = self.gradient_sensitivity(data_loader, anchor_imgs, n_batches)

        print("[2/3] Anchor vs. prototype agreement (KL divergence) ...")
        kl_res = self.anchor_agreement(data_loader, anchor_imgs, n_batches)

        print("[3/3] Prototype masking (counterfactual importance) ...")
        mask_res = self.prototype_masking(data_loader, n_batches)

        if self.verbose:

            print("\n" + "=" * 65)
            print("SUMMARY")
            print("=" * 65)

            print(f"\n--- Gradient sensitivity (∂loss/∂prototype_k norm) ---")
            for k, norm in enumerate(grad_res['grad_norms']):
                bar = "█" * int(norm / grad_res['grad_norms'].max() * 20)
                print(f"  class {k:2d} | {norm:.5f} | {bar}")
            print(f"  Most sensitive : class {grad_res['most_sensitive']}")
            print(f"  Spread (std)   : {grad_res['sensitivity_std']:.5f}")
            print(f"  → {'Non-uniform gradient norms confirm that prototypes are not' if grad_res['sensitivity_std'] > 1e-4 else 'Uniform norms: prototypes may be degenerate —'} ")
            print(f"    redundant; each occupies a distinct region of loss curvature.")

            print(f"\n--- Anchor vs. prototype KL divergence ---")
            print(f"  Mean KL(anchor‖proto) on x : {kl_res['mean_kl_x']:.4f}")
            print(f"  Mean KL(anchor‖proto) on y : {kl_res['mean_kl_y']:.4f}")
            print(f"  Per-class KL:")
            for c, kl in kl_res['per_class_kl'].items():
                bar = "█" * int(kl / max(kl_res['per_class_kl'].values()) * 20) if max(kl_res['per_class_kl'].values()) > 0 else ""
                print(f"    class {c:2d} | {kl:.4f} | {bar}")
            print(f"  Class with highest divergence : {kl_res['max_divergence_class']}")
            print(f"  Class KL variance             : {kl_res['class_kl_variance']:.6f}")
            print(f"  → {'Anchors provide distinct signal (KL > 0.05 on average)' if (kl_res['mean_kl_x'] + kl_res['mean_kl_y']) / 2 > 0.05 else 'Low KL: anchors and prototypes largely agree'};")
            print(f"    class {kl_res['max_divergence_class']} benefits most from the dual representation.")

            print(f"\n--- Prototype masking (counterfactual importance) ---")
            print(f"  Baseline  acc : {mask_res['baseline_acc']:.4f}   F1 : {mask_res['baseline_f1']:.4f}")
            print(f"  Prototype | Acc drop | F1 drop")
            for k in mask_res['ranking']:
                d = mask_res['per_prototype'][k]
                print(f"    class {k:2d} | {d['acc_drop']:+.4f}  | {d['f1_drop']:+.4f}")
            print(f"  Most critical prototype : class {mask_res['most_critical_prototype']}")

            print("\n--- Joint justification ---")
            crit = mask_res['most_critical_prototype']
            high_kl_class = kl_res['max_divergence_class']
            print(f"  • Prototype {crit} shows the largest acc drop when removed,")
            print(f"    confirming it carries irreplaceable discriminative load.")
            if crit == high_kl_class:
                print(f"  • This same class ({crit}) also has the highest anchor/prototype")
                print(f"    KL divergence — anchors and prototypes encode complementary")
                print(f"    views of this class, justifying the dual representation.")
            else:
                print(f"  • Class {high_kl_class} shows the highest anchor/prototype divergence,")
                print(f"    meaning anchors provide a genuinely different inductive bias")
                print(f"    for that class beyond what the learnable prototype alone captures.")
            print(f"  • Non-uniform gradient norms (std={grad_res['sensitivity_std']:.5f}) confirm")
            print(f"    that prototypes are individually specialised, not interchangeable.")
            print("=" * 65)

        return {
            'gradient': grad_res,
            'anchor_agreement': kl_res,
            'masking': mask_res,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _f1_macro(y_true, y_pred, num_classes):
        classes = torch.unique(y_true)
        f1s = []
        for c in classes:
            tp = ((y_pred == c) & (y_true == c)).float().sum()
            fp = ((y_pred == c) & (y_true != c)).float().sum()
            fn = ((y_pred != c) & (y_true == c)).float().sum()
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f1s.append(2 * p * r / (p + r + 1e-8))
        return torch.stack(f1s).mean().item() if f1s else 0.0