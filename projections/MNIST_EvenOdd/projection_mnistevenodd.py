import torch
import random

class Projection:
    def __init__(self):
            self.alpha = 0.0

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

    def batch_propose_neighbor(self, dx, n):
        possible_deltas = torch.tensor([-1, 1], device=dx.device)

        device = dx.device
        B = dx.shape[0]
        rand_deltas = possible_deltas[torch.randint(0, 2, (B,), device=device)]
        proposed_dx = dx + rand_deltas
        proposed_dy = n - proposed_dx

        mask_valid = (proposed_dx >= 0) & (proposed_dx <= 9) & (proposed_dy >= 0) & (proposed_dy <= 9)

        # For invalid proposals, revert to original dx and dy
        proposed_dx = torch.where(mask_valid, proposed_dx, dx)
        proposed_dy = torch.where(mask_valid, proposed_dy, n - dx)

        return torch.stack([proposed_dx, proposed_dy], dim=1)
    