import yaml
from merge_strategy import strategy

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        if 'series' in config:
            return [load_config(item)
                    for item in config['series']]
        if 'include' in config:
            includes = config['include']
            if type(includes) is not list:
                includes = [includes]
            merged = {}
            for include in includes:
                merged = strategy.merge(merged, load_config(include))
            config = strategy.merge(merged, config)
        return config