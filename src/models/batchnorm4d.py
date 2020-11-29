from torch.nn.modules.batchnorm import _BatchNorm


class BatchNorm4d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'
                             .format(input.dim()))
