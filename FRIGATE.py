import math

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
import multiprocessing as mp
import logging


def cluster_helper(labels: np.ndarray, n_clusters: int, df: pd.DataFrame):
    """
    return list of data frames of the clusters
    :param labels: labels of the samples
    :param n_clusters: number of clusters
    :param df: DataFrame of the full data
    :return: list of DataFrames, each representing a cluster
    """
    clusters = []
    for i in range(n_clusters):
        clusters.append([j for j, e in enumerate(list(map(int, labels))) if e == i])

    dfs = [df.iloc[clusters[i]] for i in range(n_clusters)]
    return dfs


def get_elbow_k(df: pd.DataFrame, max_k: int = 15):
    """
    Get the number of clusters k according to the Elbow method
    :param df: the DataFrame to cluster
    :param max_k: maximal number of clusters. Default in 15
    :return: The chosen k
    """
    distance_values = []
    for i in range(1, max_k + 1):
        kmeans_norm_i = KMeans(n_clusters=i, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(
            df)
        distance_values.append(kmeans_norm_i.inertia_)
    final_scores_dis = []
    for i in range(1, max_k - 1):
        final_scores_dis.append(distance_values[i - 1] + distance_values[i + 1] - 2 * distance_values[i])
    return np.argmax(final_scores_dis) + 2


class FRIGATE:
    """
    A class representing a Frigate object.
    """
    def __init__(self, df: pd.DataFrame, m_iterations: int = None, k_clusters: int = None, cat_cols: list = None,
                 gamma: int = 3, fixed_cols_f: float = 0.1, eta: float = 0.5, MW: bool = True, parallel: bool =False,
                 logger_level: int = logging.INFO):
        """
        Initializing a Frigate object and invokes a tun of the algorithm with the given parameters.
        :param df: a DataFrame object to be clustered, with no missing data. Rows are samples and columns are variables
        :param m_iterations: Number of iterations. Default is two times the number of variables (2*|V|)
        :param k_clusters: Number of clusters. Default is None - k_clusters will be chosen with elbow method
        :param cat_cols: List of categorical columns. Default is None. Default is 3.
        :param gamma: Weight factor for K-Prototype, if both categorical and continuous variables are present.
        :param fixed_cols_f: Fraction of columns to use in each iteration. Default is 0.1.
        :param eta: Parameter for the update rule when using Multiplicative Weight. Default is 0.5.
        :param MW: A boolean factor that determines if to use Multiplicative Weights (MW). Default is True.
        :param parallel:A boolean factor that determines if to run in parallel. Default is False.
        :param logger_level: a logging level of logger. Default is logging.INFO.
        """
        self.df = df
        self.variables = self.df.columns
        self.k = k_clusters
        self.cat_cols = cat_cols
        self.gamma = gamma
        self.fixed_cols = math.ceil(len(self.variables) * fixed_cols_f)  # number of columns to use in each iteration
        self.eta = eta
        self.M = pd.DataFrame(columns=['all_vars'], index=range(df.shape[1]))  # DataFrame penalty used for MW
        self.W = pd.DataFrame(columns=df.columns, index=['weights'])  # DataFrame of weights of all variables, for MW
        mat_vals = list(np.linspace(0, 1, len(self.variables)))  # penalty values
        self.M['all_vars'] = mat_vals

        self.W.loc['weights'] = [1] * self.W.shape[1]  # initialize weights to 1
        self.counts = pd.DataFrame(0, index=["count"], columns=self.variables)  # count the number of iteration that
                                                                                # each variable appeared in

        self.scores_dis_from_cen_per = pd.DataFrame(0, index=["scores"], columns=self.variables)  # DataFrame of scores
                                                                                                  # of all variables

        self.results = None  # A DataFrame of the ranked variables.
        self.results_with_scores = None  # A DataFrame of the ranked variables and their scores.
        self.MW = MW
        self.parallel = parallel

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logger_level, format='%(message)s')

        if m_iterations is None:
            self.m = 2 * len(df.columns)
        else:
            self.m = m_iterations

        self.frigate_algorithm()  # Invoke the Frigate algorithm.

    def get_results(self):
        return self.results

    def get_results_with_scores(self):
        return self.results_with_scores

    def get_count(self):
        return self.counts

    def update_weights(self, order, cols):
        """
        Update the weights for MW
        :param order: Ranks of the participating columns
        :param cols: Participating columns
        :return: None
        """
        for col in cols:
            i = order.index(col)
            self.W[col] = self.W[col] * np.exp(-self.eta * self.M['all_vars'][i])

    def frigate_iteration(self):
        """
        A single iteration of the algorithm, called either in parallel or sequentially
        :return: a tuple (variables, scores) of participated variables and their scores
        """
        all_cols = self.variables
        if self.MW:
            # Choosing the columns from a weights' dependent distribution
            sum_w = np.sum(self.W.loc['weights'])
            probability = self.W.loc['weights'] / sum_w
            permuted_cols = np.random.choice(list(all_cols), size=self.fixed_cols, replace=False, p=probability)
        else:
            # Choosing the columns in random
            permuted_cols = np.random.choice(list(all_cols), size=self.fixed_cols, replace=False)

        new_df = self.df[permuted_cols]
        scores_dis_from_cen_per = []  # list of scores for the participating features
        mark_array = new_df.values

        # check if we have categorical cols
        if self.cat_cols is not None:
            numeric_features_idx = [new_df.columns.get_loc(a) for a in new_df.columns if
                                    a not in self.cat_cols]
            categorical_features_idx = [new_df.columns.get_loc(a) for a in new_df.columns if
                                        a in self.cat_cols]

        is_categorical = True
        only_cat = False
        if self.cat_cols is None or len(categorical_features_idx) < 1:
            is_categorical = False
        if is_categorical:
            if len(categorical_features_idx) == len(new_df.columns):
                only_cat = True
                # If we have only categorical features we will use the KModes algorithm
                kmodes_or_proto = KModes(n_clusters=self.k, verbose=0, max_iter=400, n_init=100).fit(mark_array)
            else:
                # If we have mixed data of both categorical and continuous features we will use the KPrototype algorithm
                kmodes_or_proto = KPrototypes(n_clusters=self.k, verbose=0, max_iter=400, n_init=100,
                                              gamma=self.gamma).fit(mark_array, categorical=categorical_features_idx)
            general_score = kmodes_or_proto.cost_  # The general "solution score", which is the distance of samples to clusters' centroids
            labels = kmodes_or_proto.labels_
        else:
            # If we have only continuous features we will use the KMeans algorithm
            kmeans = KMeans(n_clusters=self.k, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(new_df)
            labels = kmeans.labels_
            general_score = kmeans.inertia_  # The general "solution score", which is the distance of samples to clusters' centroids

        dfs = cluster_helper(labels=labels, n_clusters=self.k, df=new_df)

        # If categorical features exist we separate the score to continuous and categorical for an easier update.
        if is_categorical:
            numeric_cost = 0
            categorical_cost = 0

            for i in range(len(dfs)):
                if dfs[i].empty:
                    break
                categorical_cost += np.sum(
                    np.array(kmodes_or_proto.cluster_centroids_[:, len(numeric_features_idx):][i]).reshape(1, -1) !=
                    dfs[i][dfs[i].columns[categorical_features_idx]], axis=0).sum()

                if not only_cat:
                    numeric_cost += distance.cdist(
                        np.array(kmodes_or_proto.cluster_centroids_[:, :len(numeric_features_idx)][i]).reshape(1, -1),
                        dfs[i][dfs[i].columns[numeric_features_idx]], 'sqeuclidean').sum()

        # For each feature - permute the value and calculate the new solution score
        for col in permuted_cols:

            temp_df = new_df.copy(deep=True)
            seri = pd.Series(temp_df[col], index=temp_df.index).sample(n=len(temp_df))
            seri.index = temp_df.index
            temp_df[col] = seri

            new_dfs = [[]] * len(dfs)
            for d in range(len(dfs)):
                new_dfs[d] = dfs[d].copy(deep=True)
                if dfs[d].empty:
                    break
                new_dfs[d][col] = seri
            cost = 0
            if is_categorical:
                if col in self.cat_cols:
                    for i in range(len(new_dfs)):
                        if new_dfs[i].empty:
                            break
                        cost += np.sum(np.array(kmodes_or_proto.cluster_centroids_[:, len(numeric_features_idx):][i]).
                                       reshape(1, -1) !=
                                       new_dfs[i][new_dfs[i].columns[categorical_features_idx]], axis=0).sum()
                    score_per = numeric_cost + self.gamma * cost
                else:
                    for i in range(len(new_dfs)):
                        cost += distance.cdist(
                            np.array(kmodes_or_proto.cluster_centroids_[:, :len(numeric_features_idx)][i]).reshape(1,
                                                                                                                   -1),
                            new_dfs[i][new_dfs[i].columns[numeric_features_idx]], 'sqeuclidean').sum()
                    score_per = self.gamma * categorical_cost + cost
            else:
                for i in range(len(new_dfs)):
                    cost += distance.cdist(np.array(kmeans.cluster_centers_[i]).reshape(1, -1), new_dfs[i],
                                           'sqeuclidean').sum()
                score_per = cost

            # get the score of the feature by subtracting original solution score from the new one (after permutation)
            scores_dis_from_cen_per.append(score_per - general_score)

        return permuted_cols, scores_dis_from_cen_per

    def frigate_algorithm(self):
        """
        A full run of the FRIGATE algorithm
        """

        if self.k is None:
            raise ValueError("K number of clusters must be provided")

        self.logger.info(f"k number of clusters is {self.k}")

        if self.MW and self.parallel:
            self.logger.info("Both MW and parallel are set to True. MW cannot run in parallel, the code will run "
                             "sequentially")
            self.parallel = False

        # Checks is the code should run in parallel.
        if self.parallel:
            pool = mp.Pool(int(mp.cpu_count()*0.5))
            results = [pool.apply(self.frigate_iteration_for_parallel) for p in range(self.m)]
            pool.close()
            pool.join()

            # Setting the results from the parallel run to the DataFrames
            for p in range(self.m):
                self.scores_dis_from_cen_per[results[p][0]] += results[p][1]
                self.counts[results[p][0]] += 1

        else:
            for p in range(self.m):
                results = self.frigate_iteration()

                self.scores_dis_from_cen_per[results[0]] += results[1]
                self.counts[results[0]] += 1

                if self.MW:
                    # update the weights after each iteration
                    order = list((self.scores_dis_from_cen_per / self.counts.values).sort_values(axis=1, by="scores",
                                                                                            ascending=False).columns)
                    self.update_weights(order, results[0])


        self.logger.info("After FRIGATE iterations")
        # sorting the features according to their scores
        ordered_scores_dis_from_cen_per = (self.scores_dis_from_cen_per / self.counts.values).sort_values(axis=1,
                                                                                                     by="scores",
                                                                                                     ascending=False)
        self.results = ordered_scores_dis_from_cen_per.columns
        self.results_with_scores = ordered_scores_dis_from_cen_per

