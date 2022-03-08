"""
base class
"""

from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd



class Estimator(ABC):

    def __init__(self):
        self.result = None


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

    @property
    def summary(self):
        print("=========Estimator Information=========")
        print(self.data.head(3))
        print("Recover the BN with {} variables".format(len(self.data.columns)))
        print("result:\n{}".format(self.result.adj_matrix))

    @abstractmethod
    def run(self):
        """
        run the estimator
        """

    def show(self):
        """
        Show figure of result
        :return: figure
        """
        if self.result:
            self.result.show()
        else:
            raise ValueError("No result obtained")

    def save(self, path: str):
        """

        :param path: path to save excel file
        :return: None
        """
        if self.result:
            self.result.to_excel(path)
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


    def state_count(self, x: str, parents: list):
        """
        count a multi-index table.

            r1  r2  r3
        q1
        q2

        :param x: variable name
        :param parents: list include str
        :return:
        """
        if parents:
            parents_states = [self.state_names[parent] for parent in parents]
            state_count_data = (
                self.data.groupby([x] + parents).size().unstack(parents)
            )
            row_index = self.state_names[x]
            if len(parents) == 1:
                column_index = parents_states[0]
            else:
                column_index = pd.MultiIndex.from_product(parents_states, names=parents)
            state_counts = state_count_data.reindex(index=row_index, columns=column_index).fillna(0)
        else:
            state_counts = self.data.groupby(x).size()
        return state_counts

    @abstractmethod
    def local_score(self, x, parents):
        """
        return local score
        """

    def all_score(self, dag, detail=True):
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
