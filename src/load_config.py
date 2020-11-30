import os
import yaml
from merge_strategy import strategy


def load_config(path: str):
    if os.path.isdir(path):
        # If a directory is passed, it's the same as having
        # a yaml with a 'series' list of all the files in
        # the directory.
        return [load_config(os.path.join(path, f))
                for f in os.listdir(path)]
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        if 'series' in config:
            # This is similar to passing in a directory,
            # but allows for fine-grain control over which
            # experiments are ran.
            return [load_config(item)
                    for item in config['series']]
        if 'include' in config:
            # Recursively deep merge all the includes
            includes = config['include']
            if type(includes) is not list:
                includes = [includes]
            merged = {}
            for include in includes:
                merged = strategy.merge(merged, load_config(include))
            # Merge this config file in last
            config = strategy.merge(merged, config)
        return config
