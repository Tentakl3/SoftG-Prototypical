import ltn
import torch

class Logic():
    def __init__(self):
        self.epsilon = 1e-8

    def gen_pairs(self):
        row_pairs = []
        col_pairs = []
        box_pairs = []

        for r in range(4):
            for c1 in range(4):
                for c2 in range(c1 + 1, 4):
                    row_pairs.append((r * 4 + c1, r * 4 + c2))

        for c in range(4):
            for r1 in range(4):
                for r2 in range(r1 + 1, 4):
                    col_pairs.append((r1 * 4 + c, r2 * 4 + c))

        # 2×2 boxes
        for br in range(0, 4, 2):
            for bc in range(0, 4, 2):
                cells = [(br + i) * 4 + (bc + j) for i in range(2) for j in range(2)]
                for i in range(len(cells)):
                    for j in range(i + 1, len(cells)):
                        box_pairs.append((cells[i], cells[j]))

        row_pairs = torch.tensor(row_pairs)
        col_pairs = torch.tensor(col_pairs)
        box_pairs = torch.tensor(box_pairs)

        return row_pairs, col_pairs, box_pairs

    def SudokuTruth_atomic(self, digits, boards, p):
        Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="f"
        )

        ltn_Exists = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="e"
        )

        ltn_AndProd = ltn.Connective(ltn.fuzzy_ops.AndProd())
        ltn_AndMin = ltn.Connective(ltn.fuzzy_ops.AndMin())
        ltn_OrMax = ltn.Connective(ltn.fuzzy_ops.OrMax())
        ltn_Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())

        cell = ltn.Variable("cell", boards)
        digit = ltn.Variable("digit", digits)

        def isdigit(d, c):
            board_p = torch.gather(d, -1, (c-1).unsqueeze(-1)).squeeze(-1)
            board_logp = torch.log(board_p + self.epsilon)
            return torch.exp(board_logp.mean(dim=-1))

        IsDigit = ltn.Predicate(func=isdigit)

        board_sat = Forall(ltn.diag(digit, cell), IsDigit(digit, cell)).value

        return board_sat

    def SudokuTruth(self, digits, p):
        Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="f"
        )

        ltn_Exists = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="e"
        )

        ltn_AndProd = ltn.Connective(ltn.fuzzy_ops.AndProd())
        ltn_AndMin = ltn.Connective(ltn.fuzzy_ops.AndMin())
        ltn_OrMax = ltn.Connective(ltn.fuzzy_ops.OrMax())
        ltn_Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())

        row_pairs, col_pairs, box_pairs = self.gen_pairs()

        def neq_digit(d1, d2):

            sat_pair = torch.max(d1 * d2, dim=-1).values
            #sat_pair = torch.sum(d1 * d2, dim=-1)

            return torch.mean(sat_pair, dim=1)

        NeqDigit = ltn.Predicate(func=neq_digit)

        i, j = row_pairs[:,0], row_pairs[:,1]
        d1 = digits[:, i]
        d2 = digits[:, j]

        row_d1 = ltn.Variable("r_d_1", d1)
        row_d2 = ltn.Variable("r_d_2", d2)

        i, j = col_pairs[:,0], col_pairs[:,1]
        d1 = digits[:, i]
        d2 = digits[:, j]

        col_d1 = ltn.Variable("c_d_1", d1)
        col_d2 = ltn.Variable("c_d_2", d2)

        i, j = box_pairs[:,0], box_pairs[:,1]
        d1 = digits[:, i]
        d2 = digits[:, j]

        box_d1 = ltn.Variable("b_d_1", d1)
        box_d2 = ltn.Variable("b_d_2", d2)

        row_sat = Forall(ltn.diag(row_d1, row_d2), ltn_Not(NeqDigit(row_d1, row_d2)))
        col_sat = Forall(ltn.diag(col_d1, col_d2), ltn_Not(NeqDigit(col_d1, col_d2)))
        box_sat = Forall(ltn.diag(box_d1, box_d2), ltn_Not(NeqDigit(box_d1, box_d2)))

        return ltn_AndProd(ltn_AndProd(row_sat, col_sat), box_sat).value

    def neg_SudokuTruth_features(self, digit_features, p):
        Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="f"
        )

        ltn_Exists = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="e"
        )

        ltn_AndProd = ltn.Connective(ltn.fuzzy_ops.AndProd())
        ltn_AndMin = ltn.Connective(ltn.fuzzy_ops.AndMin())
        ltn_OrMax = ltn.Connective(ltn.fuzzy_ops.OrMax())
        ltn_OrSum = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        ltn_Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())

        row_pairs, col_pairs, box_pairs = self.gen_pairs()

        def SameSymbol(z1, z2, tau=0.1):
            d = torch.norm(z1 - z2, dim=-1)
            d = torch.sum(d, dim=-1)
            return torch.exp(-d / 20)

        SameDigit = ltn.Predicate(func=SameSymbol)

        i, j = row_pairs[:,0], row_pairs[:,1]
        d1 = digit_features[:, i, :]
        d2 = digit_features[:, j, :]

        row_d1 = ltn.Variable("r_d_1", d1)
        row_d2 = ltn.Variable("r_d_2", d2)

        i, j = col_pairs[:,0], col_pairs[:,1]
        d1 = digit_features[:, i]
        d2 = digit_features[:, j]

        col_d1 = ltn.Variable("c_d_1", d1)
        col_d2 = ltn.Variable("c_d_2", d2)

        i, j = box_pairs[:,0], box_pairs[:,1]
        d1 = digit_features[:, i]
        d2 = digit_features[:, j]

        box_d1 = ltn.Variable("b_d_1", d1)
        box_d2 = ltn.Variable("b_d_2", d2)

        row_sat = Forall(ltn.diag(row_d1, row_d2), SameDigit(row_d1, row_d2))
        col_sat = Forall(ltn.diag(col_d1, col_d2), SameDigit(col_d1, col_d2))
        box_sat = Forall(ltn.diag(box_d1, box_d2), SameDigit(box_d1, box_d2))

        return ltn_OrMax(ltn_OrMax(row_sat, col_sat), box_sat).value


    def SudokuTruth_features(self, digit_features, p):
        Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="f"
        )

        ltn_Exists = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="e"
        )

        ltn_AndProd = ltn.Connective(ltn.fuzzy_ops.AndProd())
        ltn_AndMin = ltn.Connective(ltn.fuzzy_ops.AndMin())
        ltn_OrMax = ltn.Connective(ltn.fuzzy_ops.OrMax())
        ltn_Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())

        row_pairs, col_pairs, box_pairs = self.gen_pairs()

        def SameSymbol(z1, z2, tau=0.1):
            d = torch.norm(z1 - z2, dim=-1)
            d = torch.sum(d, dim=-1)
            return torch.exp(-d / 20)

        SameDigit = ltn.Predicate(func=SameSymbol)

        i, j = row_pairs[:,0], row_pairs[:,1]
        d1 = digit_features[:, i, :]
        d2 = digit_features[:, j, :]

        row_d1 = ltn.Variable("r_d_1", d1)
        row_d2 = ltn.Variable("r_d_2", d2)

        i, j = col_pairs[:,0], col_pairs[:,1]
        d1 = digit_features[:, i]
        d2 = digit_features[:, j]

        col_d1 = ltn.Variable("c_d_1", d1)
        col_d2 = ltn.Variable("c_d_2", d2)

        i, j = box_pairs[:,0], box_pairs[:,1]
        d1 = digit_features[:, i]
        d2 = digit_features[:, j]

        box_d1 = ltn.Variable("b_d_1", d1)
        box_d2 = ltn.Variable("b_d_2", d2)

        row_sat = Forall(ltn.diag(row_d1, row_d2), ltn_Not(SameDigit(row_d1, row_d2)))
        col_sat = Forall(ltn.diag(col_d1, col_d2), ltn_Not(SameDigit(col_d1, col_d2)))
        box_sat = Forall(ltn.diag(box_d1, box_d2), ltn_Not(SameDigit(box_d1, box_d2)))

        return ltn_AndMin(ltn_AndMin(row_sat, col_sat), box_sat).value