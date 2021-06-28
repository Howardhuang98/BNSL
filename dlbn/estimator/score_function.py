import numpy as np

from dlbn.base import *


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


class BicScore(Score):
    def __init__(self, contingency_table):
        """

        :param contingency_table: pd.Dataframe
        """
        super(BicScore, self).__init__()
        self.contingency_table = Data(contingency_table)

    def local_score(self, child: str, parents: list):
        values = []
        for index, series in self.contingency_table.DataFrame.iterrows():
            parents_config = series.loc[parents].to_dict()
            values.append(series['count'] * np.log(
                series['count'] / self.contingency_table.state_count_in_conTB(**parents_config)))
        return np.sum(values)


if __name__ == '__main__':
    data = pd.read_excel(r"../test/test_data.xlsx")
    a = Data(data)
    b = a.contingency_table()
    s = BicScore(b)
    print(s.local_score('A', ['B', 'C']))
