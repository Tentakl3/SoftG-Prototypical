import ltn
import torch

class Logic():
    def __init__(self):
        self.epsilon = 1e-8

    def addition_logic(self, Digit_s_d, operand_images, batch_candidates, addition_labels):
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
                ltn_AndMin(Digit_s_d(images_x, d_1), Digit_s_d(images_y, d_2)), # Use fuzzy equality
                p=2
            ).value

        return sat_agg
    
    def addition_logic(self, Digit_s_d, operand_images, batch_candidates, addition_labels):
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
                ltn_AndMin(Digit_s_d(images_x, d_1), Digit_s_d(images_y, d_2)), # Use fuzzy equality
                p=2
            ).value

        return sat_agg

    def addition_proto_logic(self, p_x, p_y, latent_digits):

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