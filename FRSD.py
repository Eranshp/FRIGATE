import random
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import math
import multiprocessing as mp


class FRSD:
    # FRSD algorithm based on k-means clustering with 100 ranking solution (M=100) and |V|/2 variables in ech clustering
    # as proposed by Hong ea al. 2007. Correlation is measured with linear correlation coefficient, as it performed
    # better than symmetrical uncertainty in Hong ea al. 2007.
    def __init__(self, df, number_of_subspaces: int = 200, fixed_cols_f: float = 0.06, k_min: int = 2, k_max: int = None,
                 parallel: bool =False):
        self.df = df
        self.variables = self.df.columns
        self.B = number_of_subspaces
        self.alpha = fixed_cols_f
        self.k_min = k_min
        self.k_max = k_max
        self.parallel = parallel
        self.counts = pd.DataFrame(0, index=["count"], columns=self.variables)  # count the number of iteration that
                                                                                # each variable appeared in
        self.results = None
        self.results_with_scores = None  # A DataFrame of the ranked variables and their scores.
        self.frsd_algorithm()

    def get_results(self):
        return self.results

    def get_results_with_scores(self):
        return self.results_with_scores

    @staticmethod
    def frsd_single_iter(df, n_clusters, count):
        """
        a single FRSD iteration
        :param df: DataFrame with reduced number of features for the iteration
        :param n_clusters: k number of clusters to use in this iteration
        :param count: count of the number of appearances of each feature
        :return: weighted scores
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(df)
        sil_score = silhouette_score(df, kmeans.labels_)
        psi = len(df.columns)
        w = (1 + sil_score) / 2  # weight factor

        scores = pd.DataFrame(columns=df.columns)
        for col in df.columns:
            temp_df = df.copy(deep=True)
            seri = pd.Series(temp_df[col], index=temp_df.index).sample(n=len(temp_df))
            seri.index = temp_df.index
            temp_df[col] = seri
            col_sil = silhouette_score(temp_df, kmeans.labels_)
            delta_col = sil_score - col_sil
            scores[col] = [delta_col]
            count[col] += 1
        scores.sort_values(axis=1, by=0, ascending=False, inplace=True)
        norm_rank = [(psi - (i)) / psi for i in range(psi)]
        scores.iloc[0] = norm_rank
        return scores * w

    def frsd_for_k(self, k, psi):
        """
        A full run for a specific k number of clusters
        :param k: the specific k nuber of clusters
        :param psi: number of features in each iteration
        :return: scores of all features for the tested k
        """
        initial_counts = [0] * len(self.df.columns)
        count = pd.DataFrame([initial_counts], columns=self.df.columns)
        scores_of_k = pd.DataFrame([initial_counts], columns=self.df.columns)
        for i in range(self.B):
            sub_space = random.sample(list(self.df.columns), k=psi)
            scores = self.frsd_single_iter(self.df[sub_space], k, count)
            for col in scores.columns:
                scores_of_k[col] += scores[col]
        final_scores_k = scores_of_k / count
        return final_scores_k

    def frsd_algorithm(self):
        print("start FRSD")
        psi = int(math.ceil(self.alpha*len(self.df.columns)))
        if not self.k_max:
            self.k_max = min(math.ceil(math.sqrt(len(self.df))), 20)
        values_k = list(range(self.k_min, self.k_max+1))
        initial_counts = [0] * len(self.df.columns)
        final_scores = pd.DataFrame([initial_counts], columns=self.df.columns)
        if self.parallel:
            pool = mp.Pool(int(mp.cpu_count()*0.5))
            results = [pool.apply(self.frsd_for_k, args=(k, psi)) for k in values_k]
            pool.close()
            pool.join()
            final_scores = sum(results)

        else:
            for k in values_k:
                count = pd.DataFrame([initial_counts], columns=self.df.columns)
                scores_of_k = pd.DataFrame([initial_counts], columns=self.df.columns)
                for i in range(self.B):
                    sub_space = np.random.permutation(self.df.columns)
                    scores = self.frsd_single_iter(self.df[sub_space[:psi]], k, count)
                    for col in scores.columns:
                        scores_of_k[col] += scores[col]
                for col in self.df.columns:
                    final_scores[col] += scores_of_k[col]/count[col]
        final_scores.loc[0] = final_scores.loc[0]/len(values_k)

        final_scores = final_scores.sort_values(axis=1, by=0, ascending=False)

        self.results = final_scores.columns
        self.results_with_scores = final_scores  # A DataFrame of the ranked variables and their scores.