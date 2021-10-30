"""
base class
"""

from abc import ABC
from abc import abstractmethod

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dlbn.direct_graph import DAG


class Estimator(ABC):

    def load_data(self, data):
        """
        加载数据
        """

        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[range(data.shape[0])])
            self.data = data
        else:
            raise ValueError("Data loading error")

    def show_est(self):
        print("=========Estimator Information=========")
        print('''
        ·▄▄▄▄    ▄▄▌    ▄▄▄▄·    ▐ ▄ 
        ██▪ ██   ██•    ▐█ ▀█▪  •█▌▐█
        ▐█· ▐█▌  ██▪    ▐█▀▀█▄  ▐█▐▐▌
        ██. ██   ▐█▌▐▌  ██▄▪▐█  ██▐█▌
        ▀▀▀▀▀•   .▀▀▀   ·▀▀▀▀   ▀▀ █▪
        ''')
        print(self.data.head(5))
        print("Recover the BN with {} variables".format(len(self.data.columns)))

    @abstractmethod
    def run(self):
        """
        run the estimator
        """

    def show(self):
        if self.result_dag:
            plt.figure()
            nx.draw_networkx(self.result_dag)
            plt.title("Bayesian network")
            plt.show()
        else:
            raise ValueError("No result obtained")


class Score(ABC):
    """
    Score base class
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.state_names = {}
        for var in list(data.columns.values):
            self.state_names[var] = sorted(list(self.data.loc[:, var].unique()))
        self.contingency_table = None

    @abstractmethod
    def local_score(self, x, parents):
        """
        return local score
        """

    def all_score(self, dag:DAG, detail=True):
        """
        return score on the DAG
        """
        score_dict = {}
        score_list = []
        for node in dag.nodes:
            parents = list(dag.predecessors(node))
            local_score = self.local_score(node, parents)
            score_list.append(local_score)
            if detail:
                score_dict[node] = local_score
        if detail:
            return sum(score_list), score_dict
        return sum(score_list)


