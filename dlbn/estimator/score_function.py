from dlbn.base import *
import numpy as np


class Score:
    def __init__(self):
        """
        Score class is base class for score functions, like BIC, MDL
        """

    def local_score(self, child, parents):
        """
        base class does not specify the score method
        """
        return 0

    def global_score(self, model: DAG):
        """
        global score for decomposable score
        """
        score = 0
        for node in model.nodes():
            score += self.local_score(node, model.predecessors(node))
        return score
