import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import multiprocessing as mp
import random
import logging


class FRMV:
    # FRMV algorithm based on k-means clustering with 100 ranking solution (M=100) and |V|/2 variables in ech clustering
    # as proposed by Hong ea al. 2007. Correlation is measured with linear correlation coefficient, as it performed
    # comparable to symmetrical uncertainty in Hong ea al. 2007.
    def __init__(self, df: pd.DataFrame, k_clusters: int, m_iterations: int = 100, fixed_cols_f: float = 0.5,
                 logger_level: int = logging.INFO, parallel: bool = False):
        """
        Initializing a Frigate object and invokes a run of the algorithm with the given parameters.
        :param df: a DataFrame object to be clustered, with no missing data. Rows are samples and columns are variables
        :param k_clusters: Number of clusters. must be provided by the user.
        :param m_iterations: Number of iterations. Default is 100
        :param fixed_cols_f: Fraction of features to use in each iteration. Default is 0.5
        :param logger_level: A logging level of logger. Default is logging.INFO.
        :param parallel: A boolean factor that determines if to run in parallel. Default is False.
        """
        self.df = df
        self.variables = self.df.columns
        self.m = m_iterations
        self.fixed_cols = math.ceil(len(self.variables) * fixed_cols_f)  # A fixed number of features to use in
                                                                         # each iteration
        self.k_clusters = k_clusters
        self.parallel = parallel  # is it a parallel run
        self.results = None
        self.results_with_scores = pd.DataFrame(index=range(self.m), columns=self.variables)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logger_level, format='%(message)s')
        self.frmv_algorithm()

    def get_results(self):
        return self.results

    def get_results_with_scores(self):
        return self.results_with_scores

    def frmv_iteration(self):
        """
        A single FRMV iteration with linear correlation coefficient
        :return: a sorted list of features, by their scores, and a list of ranks accordingly.
        """
        all_cols = self.variables
        permuted_cols = random.choices(list(all_cols), k=self.fixed_cols)
        new_df = self.df[permuted_cols]
        kmeans = KMeans(n_clusters=self.k_clusters, random_state=0, max_iter=400, n_init=100).fit(new_df)
        labels = kmeans.labels_

        new_df = new_df.loc[:, ~new_df.columns.duplicated()].copy()
        set_permuted_cols = new_df.columns
        c_c_df = pd.DataFrame(columns=set_permuted_cols)
        # Calculate the linear correlation coefficient
        for var in set_permuted_cols:
            c_c = np.cov(new_df[var], labels)[0][1] / (np.std(new_df[var]) * np.std(labels))
            c_c_df[var] = [c_c]
        c_c_df.sort_values(axis=1, by=0, ascending=False, inplace=True)
        return c_c_df.columns, list(range(len(c_c_df.columns)))

    def frmv_algorithm(self):
        self.logger.info("start frmv")
        if self.parallel:
            pool = mp.Pool(int(mp.cpu_count()*0.5))
            results = [pool.apply(self.frmv_iteration, args=[k]) for k in range(self.m)]
            pool.close()
            pool.join()
            for k in range(self.m):
                self.results_with_scores.iloc[k][results[k][0]] = results[k][1]
        else:
            for k in range(self.m):
                all_cols = self.variables
                permuted_cols = np.random.choice(list(all_cols), size=self.fixed_cols, replace=True)
                new_df = self.df[permuted_cols]
                kmeans = KMeans(n_clusters=self.k_clusters, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(new_df)
                labels = kmeans.labels_

                new_df = new_df.loc[:, ~new_df.columns.duplicated()].copy()
                set_permuted_cols = new_df.columns
                c_c_df = pd.DataFrame(columns=set_permuted_cols)
                for var in set_permuted_cols:
                    c_c = np.cov(new_df[var], labels)[0][1] / (np.std(new_df[var]) * np.std(labels))
                    c_c_df[var] = [c_c]
                c_c_df.sort_values(axis=1, by=0, ascending=False, inplace=True)
                self.results_with_scores.iloc[k][c_c_df.columns] = list(range(len(c_c_df.columns)))

        self.results = self.results_with_scores.mean().sort_values().index.values