import numpy as np
import pandas as pd
from FRCM import FRCM
from FRMV import FRMV
from FRSD import FRSD
from FRIGATE import FRIGATE
from FRIGATE import get_elbow_k
import logging


def build_numeric_simulation(mu, sigma, number_of_informative_vars, number_of_noise_vars,
                             number_of_samples_in_each_cluster, k_clusters):
    """
    Creates a DataFrame of simulated numeric values
    :param mu: parameter for the mean of the multivariate normal distribution
    :param sigma: parameter for the std of the multivariate normal distribution
    :param number_of_informative_vars: number of variables that differ between clusters
    :param number_of_noise_vars: number of variables that don't differ between clusters
    :param number_of_samples_in_each_cluster: number of samples in each_cluster
    :param k_clusters: number of clusters
    :return: Creates a DataFrame of simulated numeric values
    """
    mu, sigma = mu, sigma  # mean and standard deviation

    # The covariances matrix
    cov = (1 - sigma) * np.identity(number_of_informative_vars + number_of_noise_vars) + sigma * (
        np.matrix([1] * (number_of_informative_vars + number_of_noise_vars)).T) * (
              np.matrix([1] * (number_of_informative_vars + number_of_noise_vars)))

    arr_of_data_to_concat = []

    # create data with different mu values for each cluster
    for i in range(k_clusters):
        ci = np.random.multivariate_normal([0] * number_of_noise_vars + [i * mu] * number_of_informative_vars, cov,
                                           number_of_samples_in_each_cluster)
        arr_of_data_to_concat.append(ci)

    sim_df = pd.DataFrame(np.concatenate(arr_of_data_to_concat, axis=0))
    return sim_df


def build_categorical_simulation(df_cat, number_of_informative_vars, number_of_cat_vars, k_clusters, cat_epsilon,
                                 number_of_samples_in_each_cluster):
    """
    Alter the input variable df_cat into a DataFrame with simulated categorical data
    :param df_cat: empty DataFrame in the right shape for the simulated categorical data
    :param number_of_informative_vars: number of informative categorical variables
    :param number_of_cat_vars: total number of categorical variables
    :param k_clusters: k number of clusters
    :param cat_epsilon: the probability of choosing the wrong value
    :param number_of_samples_in_each_cluster: number of samples in each cluster
    :return: None
    """
    for i in range(number_of_informative_vars):
        informative_cat = []
        for k in range(k_clusters):
            p = [cat_epsilon / (k_clusters - 1)] * k_clusters
            p[k] = 1 - cat_epsilon
            informative_cat += list(
                np.random.choice(list(range(k_clusters)), size=number_of_samples_in_each_cluster, replace=True, p=p))
        df_cat.iloc[:, i] = informative_cat
    for i in range(number_of_cat_vars - number_of_informative_vars):
        p = [1 / k_clusters] * k_clusters
        df_cat.iloc[:, number_of_informative_vars + i] = list(
            np.random.choice(list(range(k_clusters)), size=number_of_samples_in_each_cluster * k_clusters, replace=True, p=p))


def z_normalization(df):
    """
    Transforming the input DafaFrame to be z-score normalized
    :param df: DataFrame of continuous data
    :return: None
    """
    for col in df.columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()


def full_simulation(logger, sigma, mu, gamma=2, number_of_informative_num_vars=20, number_of_noise_num_vars=80,
                    number_of_informative_cat_vars=20, number_of_noise_cat_vars=80, use_cat=False, is_normalized=True,
                    k_clusters=4, number_of_samples_in_each_cluster=50, cat_epsilon=0.05, paral=True, path=''):
    """
    Creates simulated data and test the performance of five algorithms: Frigate, Frigate with MW, FRCM, FRSD, FRMV.
    The measurement is the fraction true informative features in the top "number_of_informative_vars" in the final
    ranking of each algorithm.
    :param logger: A logger object.
    :param gamma: A weighing parameter when categorical data in presented.
    :param sigma: A std parameter for the multivariate normal distribution
    :param mu: A mean parameter for the multivariate normal distribution
    :param number_of_informative_num_vars: number of informative continuous variables
    :param number_of_noise_num_vars: number of non-informative continuous variables
    :param number_of_informative_cat_vars: number of informative categorical variables
    :param number_of_noise_cat_vars: number of non-informative categorical variables
    :param use_cat: Boolean parameter for using categorical variables, or only continuous
    :param is_normalized: Boolean parameter for performing z-score normalization on the continuous data
    :param k_clusters: k number of clusters
    :param number_of_samples_in_each_cluster: number of samples in each cluster
    :param cat_epsilon: parameter for creating the categorical simulation. account for the probability of choosing a
    wrong value for the informative features.
    :param paral: Boolean parameter for running in parallel
    :param path: the path to save the results
    :return: None
    """
    logger.info(f"number of samples in cluster:{number_of_samples_in_each_cluster}")
    logger.info(f"number of informative numeric variables:{number_of_informative_num_vars}")
    logger.info(f"number of noise numeric variables:{number_of_noise_num_vars}")
    logger.info(f"use categorical features = {use_cat}")
    logger.info(f"number of informative catigorical variables:{number_of_informative_cat_vars}")
    logger.info(f"number of noise catigorical variables:{number_of_noise_cat_vars}")
    logger.info(f"gamma = {gamma}")
    logger.info(f"categorical epsilon is {cat_epsilon}")

    logger.info(f"number k of clusters = {k_clusters}")
    logger.info(f"normalize data = {is_normalized}")
    logger.info(f"is parallel = {paral}")
    logger.info(f"sigma is {sigma}")
    logger.info(f"mu is {mu}")

    number_of_num_vars = number_of_informative_num_vars+number_of_noise_num_vars
    number_of_cat_vars = number_of_informative_cat_vars+number_of_noise_cat_vars

    number_of_informative_vars = number_of_informative_num_vars + number_of_informative_cat_vars if use_cat \
        else number_of_informative_num_vars

    frigate_scores_num = []
    frigate_scores_cat = []

    frigate_mw_scores_num = []
    frigate_mw_scores_cat = []

    frsd_scores = []
    frcm_scores = []
    frmv_scores = []

    # generate 10 solution for each set of parameters
    for n in range(10):
        df_num = build_numeric_simulation(mu, sigma, number_of_informative_num_vars, number_of_noise_num_vars,
                                          number_of_samples_in_each_cluster, k_clusters)
        if is_normalized:
            z_normalization(df_num)

        # k is an input for Frigate, Frigate with MW and FRMV. We produce it with Elbow method.
        k_frigate = get_elbow_k(df_num)

        cat_cols = None
        if use_cat:
            df_cat = pd.DataFrame(columns=range(len(df_num.columns), len(df_num.columns) + number_of_cat_vars),
                                  index=range(k_clusters * number_of_samples_in_each_cluster))
            build_categorical_simulation(df_cat, number_of_informative_cat_vars, number_of_cat_vars, k_clusters,
                                         cat_epsilon,
                                         number_of_samples_in_each_cluster)
            cat_cols = df_cat.columns
            df_num_cat = pd.DataFrame(np.concatenate([df_num, df_cat], axis=1))
        else:
            df_num_cat = df_num

        if not use_cat:
            # when we don't use categorical features we test all algorithms.
            frmv_obj = FRMV(df=df_num_cat, k_clusters=k_frigate, parallel=paral)
            results_frmv = frmv_obj.get_results()
            # Test the fraction of top features in the solution that are truly informative.
            results_frmv = [1 for x in results_frmv[:(number_of_informative_num_vars)] if x in list(
                range(number_of_noise_num_vars, number_of_num_vars))]
            results_frmv = np.sum(results_frmv) / number_of_informative_num_vars
            logger.debug(f"frmv result is {results_frmv}")
            frmv_scores.append(results_frmv)

            frcm_obj = FRCM(df_num_cat, parallel=paral)
            frcm_results = frcm_obj.get_results()
            frcm_results = [1 for x in frcm_results[:(number_of_informative_num_vars)] if x in list(
                range(number_of_noise_num_vars, number_of_num_vars))]
            frcm_results = np.sum(frcm_results) / number_of_informative_num_vars
            logger.debug(f"frcm result is {frcm_results}")
            frcm_scores.append(frcm_results)

            frsd_obj = FRSD(df_num_cat, parallel=paral)
            frsd_results = frsd_obj.get_results()
            frsd_results = [1 for x in frsd_results[:(number_of_informative_num_vars)] if x in list(
                range(number_of_noise_num_vars, number_of_num_vars))]
            frsd_results = np.sum(frsd_results) / number_of_informative_num_vars
            logger.debug(f"frsd result is {frsd_results}")
            frsd_scores.append(frsd_results)

        # the frigate algorithms are tested on both types of data.
        FRIGATE_obj = FRIGATE(df=df_num_cat, k_clusters=k_frigate, MW=False, cat_cols=cat_cols, parallel=paral, m_iterations=10)
        results_FRIGATE = FRIGATE_obj.get_results()
        results_FRIGATE_num = [1 for x in results_FRIGATE[:(number_of_informative_vars)] if x in list(
            range(number_of_noise_num_vars, number_of_num_vars))]
        results_FRIGATE_num = np.sum(results_FRIGATE_num) / number_of_informative_num_vars
        logger.debug(f"frigate num result is {results_FRIGATE_num}")
        frigate_scores_num.append(results_FRIGATE_num)
        if use_cat:
            results_FRIGATE_cat = [1 for x in results_FRIGATE[:(number_of_informative_vars)] if x in
                                   list(range(number_of_num_vars, number_of_num_vars + number_of_informative_cat_vars))]
            results_frigate_cat = np.sum(results_FRIGATE_cat) / number_of_informative_cat_vars
            logger.debug(f"results frigate cat = {results_frigate_cat}")
            frigate_scores_cat.append(results_frigate_cat)

        FRIGATE_MW_obj = FRIGATE(df=df_num_cat, k_clusters=k_frigate, MW=True, cat_cols=cat_cols, parallel=paral, m_iterations=10)
        results_FRIGATE = FRIGATE_MW_obj.get_results()
        results_FRIGATE_MW_num = [1 for x in results_FRIGATE[:(number_of_informative_vars)] if x in list(
            range(number_of_noise_num_vars, number_of_num_vars))]
        results_FRIGATE_MW_num = np.sum(results_FRIGATE_MW_num) / number_of_informative_num_vars
        logger.debug(f"frigate num result is {results_FRIGATE_MW_num}")
        frigate_mw_scores_num.append(results_FRIGATE_MW_num)
        if use_cat:
            results_FRIGATE_MW_cat = [1 for x in results_FRIGATE[:(number_of_informative_vars)] if x in
                                   list(range(number_of_num_vars, number_of_num_vars + number_of_informative_cat_vars))]
            results_frigate_mw_cat = np.sum(results_FRIGATE_MW_cat) / number_of_informative_cat_vars
            logger.debug(f"results frigate mw cat = {results_frigate_mw_cat}")
            frigate_mw_scores_cat.append(results_frigate_mw_cat)

    # Create a DataFrame with the mean and std of the results of each algorithm.
    df_results = pd.DataFrame(columns=["FRIGATE", "FRIGATE with MW", "FRMV", "FRCM", "FRSD"], index=["mean", "std"])

    logger.info("frigate_scores_num")
    logger.info(np.mean(frigate_scores_num))
    logger.info(np.std(frigate_scores_num))
    df_results.loc["mean"]["FRIGATE"] = np.mean(frigate_scores_num)
    df_results.loc["std"]["FRIGATE"] = np.std(frigate_scores_num)

    logger.info("frigate_mw_scores_num")
    logger.info(np.mean(frigate_mw_scores_num))
    logger.info(np.std(frigate_mw_scores_num))
    df_results.loc["mean"]["FRIGATE with MW"] = np.mean(frigate_mw_scores_num)
    df_results.loc["std"]["FRIGATE with MW"] = np.std(frigate_mw_scores_num)

    if use_cat:
        logger.info("frigate_scores_cat")
        logger.info(np.mean(frigate_scores_cat))
        logger.info(np.std(frigate_scores_cat))

        logger.info("frigate_mw_scores_cat")
        logger.info(np.mean(frigate_mw_scores_cat))
        logger.info(np.std(frigate_mw_scores_cat))

    logger.info("frmv")
    logger.info(np.mean(frmv_scores))
    logger.info(np.std(frmv_scores))
    df_results.loc["mean"]["FRMV"] = np.mean(frmv_scores)
    df_results.loc["std"]["FRMV"] = np.std(frmv_scores)

    logger.info("frcm")
    logger.info(np.mean(frcm_scores))
    logger.info(np.std(frcm_scores))
    df_results.loc["mean"]["FRCM"] = np.mean(frcm_scores)
    df_results.loc["std"]["FRCM"] = np.std(frcm_scores)

    logger.info("frsd")
    logger.info(np.mean(frsd_scores))
    logger.info(np.std(frsd_scores))
    df_results.loc["mean"]["FRSD"] = np.mean(frsd_scores)
    df_results.loc["std"]["FRSD"] = np.std(frsd_scores)

    # saves the results to a file
    file_name = f'{path}simulation_results.mu-{mu},sigma-{sigma},use_cat-{use_cat},k_clusters-{k_clusters},'\
                          f'is_normalized-{is_normalized}.csv'
    df_results.to_csv(file_name)
    return


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    # #not in use now - can add another loop for testing values of gamma
    # gamma_list = [0.5, 1, 2, 3, 5, 10]

    mu_list = [1, 2, 4]
    sigma_list = [0, 0.05, 0.2]
    normalized_list = [True, False]
    k_list = [2, 4]
    parallel=False


    for norm in normalized_list:
        for mu in mu_list:
            for sigma in sigma_list:
                for k in k_list:
                    full_simulation(logger,
                                    mu=mu,
                                    sigma=sigma,
                                    use_cat=False,
                                    k_clusters=k, number_of_samples_in_each_cluster=50,
                                    is_normalized=norm,
                                    paral=parallel)
