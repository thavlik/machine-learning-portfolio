import ray
from ray import tune
from ray.tune.integration.torch import DistributedTrainableCreator
from ray.util.sgd.torch import is_distributed_trainable

ray.init()

def my_trainable(config, checkpoint_dir=None):
    if is_distributed_trainable():
        pass

trainable = DistributedTrainableCreator(
    my_trainable,
    use_gpu=True,
    num_workers=4,
    num_cpus_per_worker=1,
)

config = {}

tune.run(
    trainable,
    resources_per_trial=None,
    config=config,
)