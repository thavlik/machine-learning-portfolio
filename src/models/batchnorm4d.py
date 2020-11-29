from torch.nn import BatchNorm3d


class BatchNorm4d(BatchNorm3d):
    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'
                             .format(input.dim()))
