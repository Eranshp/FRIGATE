import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
import multiprocessing as mp
import random


def make_aff_mats(df, df_dis_pairs, var, variables):
    """
    Make an affinity matrix for a given variable. To use when running multiple runs on the same data.
    :param df: DataFrame of the data.
    :param df_dis_pairs: DataFrame of distance matrices.
    :param var: string of the name of the variable to make the matrix for.
    :param variables: list of variables.
    :return: DataFrame of the upper triangle of the affinity matrix.
    """
    affinity_mat = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            score = math.sqrt(1 - (((df[variables[var]].iloc[i] - df[variables[var]].iloc[j]) ** 2) /
                                   df_dis_pairs.iloc[i].iloc[j]))
            affinity_mat[i][j] = score
            affinity_mat[j][i] = score
    upper_affinity_mat = np.triu(affinity_mat, k=1)
    return upper_affinity_mat


def co_association_mat(labels, k):
    """
    Make a co-association matrix
    :param labels: list of labels of a clustering results
    :param k: number of clusters
    :return: a DataFrame of the co-association matrix for the clustering solution
    """
    lists = []  # list of indexes of samples according to cluster membership
    for i in range(k):
        lists.append([j for j, x in enumerate(labels) if x == int(i)])
    co_association_mat = np.zeros((len(labels), len(labels)))
    for lst in lists:
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                co_association_mat[lst[i]][lst[j]] = 1
                co_association_mat[lst[j]][lst[i]] = 1
    return co_association_mat


class FRCM:
    """
    A class representing an FRCM object, a feature ranking for clustering algorithm proposed by Zhang et al. 2012. This 
    algorithm is composed of two part: 1) creating multiple clustering solutions, 2) evaluating the contribution of each 
    feature according to an adjustment of ARI score between the consensus matrix of the clustering solutions and an 
    affinity matrix created for each feature.  
    """
    def __init__(self, df: pd.DataFrame, t_iterations: int = 100, k_min: int = 2, k_max: int = 20,
                 parallel: bool = False, upper_aff_mats: list = None):
        """
        Initializing an FRCM object and invokes a run of the algorithm with the given parameters.
        :param df: a DataFrame object to be clustered, with no missing data. Rows are samples and columns are variables
        :param t_iterations: Number of iterations. Default is 100
        :param k_min: Minimal number of clusters. Default is 2
        :param k_max: Maximal number of clusters. Default is 20
        :param parallel:A boolean factor that determines if to run in parallel. Default is False.
        :param logger_level: a logging level of logger. Default is logging.INFO.
        :param upper_aff_mats: Upper triangle of affinity matrices that represents the static relations between features
        default is None and will be calculated during the FRCM run.
        """

        self.df = df
        self.variables = self.df.columns
        self.k_min = k_min
        self.k_max = k_max
        self.results = None  # A DataFrame of the ranked variables.
        self.results_with_scores = None  # A DataFrame of the ranked variables and their scores.
        self.parallel = parallel

        self.T = t_iterations
        self.upper_aff_mats = upper_aff_mats

        self.frcm_algorithm()  # Invoke the FRCM algorithm.

    def get_results(self):
        return self.results

    def get_results_with_scores(self):
        return self.results_with_scores

    def frcm_iteration(self, variables, list_of_k):
        """
        A single iteration of the FRCM algorithm that produces a clustering solution with half the variables
        :param variables: list of features
        :param list_of_k: list of possible k number of clusters
        :return: the co-association matrix for the clustering solution
        """
        subset = random.sample(variables, int(len(variables) / 2))
        k = random.sample(list_of_k, 1)[0]
        kmeans = KMeans(n_clusters=k, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(self.df[subset])
        co_ass_mat = co_association_mat(kmeans.labels_, k)
        return co_ass_mat

    def final_scores_calc(self, variables, var, df_dis_pairs, upper_consensus_mat):
        """
        Calculate the final score of a variable, based on a modified ARI score.
        :param variables: list of features
        :param var: the variable to calculate the score for
        :param df_dis_pairs: DataFrame of pairwise distances
        :param upper_consensus_mat: upper triangle of the consensus matrix
        :return: ARImm Score
        """

        if self.upper_aff_mats is None:
            affinity_mat = np.zeros((len(self.df), len(self.df)))
            for i in range(len(self.df)):
                for j in range(i + 1, len(self.df)):
                    score = math.sqrt(1 - (((self.df[variables[var]].iloc[i] - self.df[variables[var]].iloc[j]) ** 2) /
                                           df_dis_pairs.iloc[i].iloc[j]))
                    affinity_mat[i][j] = score
                    affinity_mat[j][i] = score
            upper_affinity_mat = np.triu(affinity_mat, k=1)
        else:
            upper_affinity_mat = self.upper_aff_mats[var]
        s0 = (np.multiply(upper_consensus_mat, upper_affinity_mat)).sum()
        s1 = upper_consensus_mat.sum()
        s2 = upper_affinity_mat.sum()
        s3 = s1 * s2 / (len(self.df) * (len(self.df) - 1))

        ARImm = (s0 - s3) / (0.5 * (s1 + s2) - s3)
        return ARImm

    def frcm_algorithm(self):
        """
        a full FRCM run
        :return:
        """
        print("start FRCM")
        self.k_max = min(math.ceil(math.sqrt(len(self.df))), self.k_max)
        list_of_k = list(range(self.k_min, self.k_max+1))
        variables = list(self.df.columns)
        if self.parallel:  # run in parallel
            pool = mp.Pool(int(mp.cpu_count()*0.5))
            args = (variables, list_of_k)
            results = [pool.apply(self.frcm_iteration, args=args) for i in range(self.T)]
            pool.close()
            pool.join()
            consensus_mat = sum(results)

        else:  # run sequentially
            t = 0
            consensus_mat = np.zeros((len(self.df), len(self.df)))
            while t < self.T:
                mat = self.frcm_iteration(variables, list_of_k)
                consensus_mat += mat
                t += 1
        consensus_mat = consensus_mat/self.T
        upper_consensus_mat = np.triu(consensus_mat, k=1)

        print("The production of clustering solutions in FRCM is done. calculating the features' scores")

        z_scores = pd.DataFrame(columns=variables, index=[0])
        df_dis_pairs = None
        if not self.upper_aff_mats:
            df_dis_pairs = pd.DataFrame(columns=self.df.index, index=self.df.index)
            for i in range(len(self.df)):
                for j in range(i + 1, len(self.df)):
                    df_dis_pairs.iloc[i].iloc[j] = (np.linalg.norm(self.df.iloc[i]-self.df.iloc[j]))**2

        if self.parallel:
            pool = mp.Pool(int(mp.cpu_count()*0.5))
            results = [pool.apply(self.final_scores_calc, args=[variables, var,
                                         df_dis_pairs, upper_consensus_mat]) for var in range(len(variables))]
            pool.close()
            pool.join()
            z_scores.iloc[0] = results
        else:
            for var in range(len(variables)):
                ARImm = self.final_scores_calc(variables, var, df_dis_pairs, upper_consensus_mat)
                z_scores[variables[var]] = [ARImm]
        ordered_scores = z_scores.sort_values(axis=1, by=0, ascending=False)
        self.results = ordered_scores.columns
        self.results_with_scores = ordered_scores

