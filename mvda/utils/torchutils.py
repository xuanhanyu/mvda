import torch


class TensorUser:

    def _tensorize_(self, mat):
        return mat if torch.is_tensor(mat) else torch.tensor(mat)
