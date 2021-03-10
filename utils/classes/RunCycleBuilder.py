from collections import namedtuple
from itertools import product


class RunCycleBuilder:

    @staticmethod
    def get_runs(params) -> list:

        Run = namedtuple("Run", params.keys())

        runs = []

        for v in product(*params.values()):
            runs.append(Run(*v))
            
        return runs
