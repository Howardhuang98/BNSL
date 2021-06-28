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


class BicScore(Score):
    def __init__(self, contingency_table):
        super(BicScore, self).__init__()
        self.contingency_table = Data(contingency_table)

    def local_score(self, child: str, parents: list):
        child_states = self.contingency_table.state_names[child]
        var_cardinality = len(child_states)
        state_counts = self.contingency_table.DataFrame.values
        sample_size = pd.sum(self.contingency_table['count'])
        num_parents_states = self.contingency_table.collect_state_names(parents)

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=np.float_)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=np.float_)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        score = np.sum(log_likelihoods)
        score -= 0.5 * np.log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score
