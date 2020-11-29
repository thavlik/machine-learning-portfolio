from deepmerge import Merger

strategy = Merger([(list, "override"),
                   (dict, "merge")],
                  ["override"],
                  ["override"])