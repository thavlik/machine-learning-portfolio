from deepmerge import Merger

strategy = Merger([(list, "override"),
                   (dict, "merge")],
                  ["override"],
                  ["override"])
            
def deep_merge(base: dict, next_: dict) -> dict:
    return strategy.merge(base, next_)
