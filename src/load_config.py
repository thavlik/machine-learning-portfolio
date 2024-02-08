import os
import yaml
from typing import List, Union

from merge_strategy import deep_merge


def load_config(path: Union[str, List[str]]) -> Union[dict, List[dict]]:
    paths = path if type(path) == list else [path]
    result = {}
    for path in paths:
        if os.path.isdir(path):
            # If a directory is passed, it's the same as having
            # a yaml with a 'series' list of all the files in
            # the directory.
            configs = []
            for f in os.listdir(path):
                if os.path.basename(f) == 'include':
                    # Include folders do not contain any files
                    # that can be executed directly.
                    continue
                fp = os.path.join(path, f)
                if not os.path.isdir(fp) and not f.endswith('.yaml'):
                    # Ignore non-yaml files like README.md
                    continue
                fc = load_config(fp)
                if type(fc) == list and len(fc) == 0:
                    # Exclude empty directories
                    continue
                configs.append(fc)
            return configs
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        if 'include' in config:
            # Recursively deep merge all the includes
            includes = config['include']
            if type(includes) is not list:
                includes = [includes]
            merged = {}
            for include in includes:
                merged = deep_merge(merged, load_config(include))
            # Merge this config file in last
            config = deep_merge(merged, config)
            # Remove include directive now that merge has occured
            del config['include']
        result = deep_merge(result, config)
    return result
