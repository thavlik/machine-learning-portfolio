from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmup(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 lr: float,
                 num_steps: int,
                 *args,
                 **kwargs):
        self._lr = lr
        self._num_steps = num_steps
        super().__init__(optimizer, *args, **kwargs)

    def get_lr(self):
        lr_scale = min(1.0, float(self._step_count + 1) /
                       float(self._num_steps))
        lr = lr_scale * self._lr
        return [lr] * len(self.optimizer.param_groups)
