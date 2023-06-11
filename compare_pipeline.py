import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
import multiprocessing as mp
import logging
from FRIGATE import FRIGATE
from FRIGATE import get_elbow_k
from FRMV import FRMV
from FRCM import FRCM
from FRCM import make_aff_mats
from FRSD import FRSD


def compare_performance(data, df_numeric, lables, nunber_of_features, gamma=3, cat_cols = None, k_clusters=2, name = "",
            path="", title="", include_cat = False, paral=True):

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    logger.info(f"run in parallel = {paral}")

    if not include_cat:
        data = df_numeric

    FRIGATE_k = get_elbow_k(df_numeric)
    logger.info(f"k number of clusters by elbow method = {FRIGATE_k}")

    FRIGATE_ari = []
    FRIGATE_MW_ari = []
    FRSD_ari = []
    FRCM_ari = []
    FRMV_ari = []
    for i in range(len(nunber_of_features)):
        FRIGATE_ari.append([])
        FRIGATE_MW_ari.append([])
        FRSD_ari.append([])
        FRMV_ari.append([])
        FRCM_ari.append([])

    variables = data.columns

    upper_aff_mats = None  # needed for FRCM
    if not include_cat:
        # make the affinity matrices for FRCM so it will only be produced once
        df_dis_pairs = pd.DataFrame(columns=data.index, index=data.index)
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                df_dis_pairs.iloc[i].iloc[j] = (np.linalg.norm(data.iloc[i] - data.iloc[j])) ** 2

        if paral:
            pool = mp.Pool(int(mp.cpu_count()*0.5))
            results = [pool.apply(make_aff_mats, args=[data, df_dis_pairs, var, variables]) for var in range(len(variables))]
            pool.close()
            pool.join()
            upper_aff_mats = results

        else:
            upper_aff_mats = []
            for var in range(len(variables)):
                upper_aff_mats.append(make_aff_mats(data, df_dis_pairs, var, variables))

    # make 10 runs of each algorithm and calculate the results
    for i in range(10):

        FRIGATE_MW_obj = FRIGATE(df=data, k_clusters=FRIGATE_k, MW=False, parallel=paral)
        results_FRIGATE = FRIGATE_MW_obj.get_results()

        FRIGATE_MW_obj = FRIGATE(df=data, k_clusters=FRIGATE_k, MW=True, parallel=paral)
        results_FRIGATE_MW = FRIGATE_MW_obj.get_results()

        if not include_cat:
            frmv_obj = FRMV(df=data,
                            k_clusters=FRIGATE_k, parallel=paral)
            frmv_run = frmv_obj.get_results()

            frcm_obj = FRCM( data, upper_aff_mats=upper_aff_mats, parallel=paral)
            frcm_run = frcm_obj.get_results()

            frsd_obj = FRSD(data, parallel=paral)
            frsd_run = frsd_obj.get_results()

        # for each run of each algorithm calculate the ARI for growing number of features, according to rank.
        for n in range(len(nunber_of_features)):

            # FRIGATE with MW
            subset_fraction = results_FRIGATE_MW[:nunber_of_features[n]]
            mark_array = data[subset_fraction].values

            categorical_features_idx = [subset_fraction.get_loc(a) for a in subset_fraction if a in cat_cols]

            # Check which is the appropriate clustering algorithm, in case of presence of categorical variables
            if len(categorical_features_idx) == len(subset_fraction):
                kmodes = KModes(n_clusters=k_clusters, verbose=0, max_iter=100, n_init=50).fit(mark_array)
                run_labels = kmodes.labels_
            elif len(categorical_features_idx) == 0:
                kmeans = KMeans(n_clusters=k_clusters, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(
                    data[subset_fraction])
                run_labels = kmeans.labels_
            else:
                kproto = KPrototypes(n_clusters=k_clusters, verbose=0, max_iter=100, n_init=50, gamma=gamma).fit(
                    mark_array,
                    categorical=categorical_features_idx)
                run_labels = kproto.labels_

            ARI = adjusted_rand_score(list(run_labels), lables)
            FRIGATE_MW_ari[n].append(ARI)

            # FRIGATE

            subset_fraction = results_FRIGATE[:nunber_of_features[n]]

            mark_array = data[subset_fraction].values
            categorical_features_idx = [subset_fraction.get_loc(a) for a in subset_fraction if a in cat_cols]

            # Check which is the appropriate clustering algorithm, in case of presence of categorical variables
            if len(categorical_features_idx) == len(subset_fraction):
                kmodes = KModes(n_clusters=k_clusters, verbose=0, max_iter=100, n_init=50).fit(mark_array)
                run_labels = kmodes.labels_
            elif len(categorical_features_idx) == 0:
                kmeans = KMeans(n_clusters=k_clusters, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(
                    data[subset_fraction])
                run_labels = kmeans.labels_
            else:
                kproto = KPrototypes(n_clusters=k_clusters, verbose=0, max_iter=100, n_init=50, gamma=gamma).fit(
                    mark_array,
                    categorical=categorical_features_idx)
                run_labels = kproto.labels_

            ARI = adjusted_rand_score(list(run_labels), lables)
            FRIGATE_ari[n].append(ARI)

            if not include_cat:

                subset_fraction = frmv_run[:nunber_of_features[n]]
                kmeans = KMeans(n_clusters=k_clusters, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(data[subset_fraction])
                ARI = adjusted_rand_score(list(kmeans.labels_), lables)
                FRMV_ari[n].append(ARI)

                subset_fraction = frsd_run[:nunber_of_features[n]]
                kmeans = KMeans(n_clusters=k_clusters, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(data[subset_fraction])
                ARI = adjusted_rand_score(list(kmeans.labels_), lables)
                FRSD_ari[n].append(ARI)
                if FRCM:
                    subset_fraction = frcm_run[:nunber_of_features[n]]
                    kmeans = KMeans(n_clusters=k_clusters, random_state=0, max_iter=400, n_init=100, init='k-means++').fit(data[subset_fraction])
                    ARI = adjusted_rand_score(list(kmeans.labels_), lables)
                    FRCM_ari[n].append(ARI)

    # calculate the average and the std of each algorithm
    shacle_no_MW_ave = []
    shacle_no_MW_std = []
    for arr in FRIGATE_ari:
        shacle_no_MW_ave.append(np.mean(arr))
        shacle_no_MW_std.append(np.std(arr))


    shacle_05_ave = []
    shacle_05_std = []
    for arr in FRIGATE_MW_ari:
        shacle_05_ave.append(np.mean(arr))
        shacle_05_std.append(np.std(arr))

    if not include_cat:

        frmv_results_ave = []
        frmv_results_std = []
        for arr in FRMV_ari:
            frmv_results_ave.append(np.mean(arr))
            frmv_results_std.append(np.std(arr))


        frsd_results_ave = []
        frsd_results_std = []
        for arr in FRSD_ari:
            frsd_results_ave.append(np.mean(arr))
            frsd_results_std.append(np.std(arr))

        if FRCM:
            frcm_results_ave = []
            frcm_results_std = []
            for arr in FRCM_ari:
                frcm_results_ave.append(np.mean(arr))
                frcm_results_std.append(np.std(arr))

    plt.title(title)

    plt.plot(nunber_of_features, shacle_no_MW_ave,
                 label=f"FRIGATE", linestyle="-", linewidth=4, color="tab:blue")
    plt.fill_between(nunber_of_features, shacle_no_MW_ave - shacle_no_MW_std, shacle_no_MW_ave + shacle_no_MW_std
                     , color="tab:blue", alpha=0.4)

    plt.plot(nunber_of_features, shacle_05_ave,
                 label=f"FRIGATE with MW", linestyle="-", linewidth=3.25, color="tab:orange")
    plt.fill_between(nunber_of_features, shacle_05_ave - shacle_05_std, shacle_05_ave + shacle_05_std
                     , color="tab:orange", alpha=0.4)


    if not include_cat:
        plt.plot(nunber_of_features, frcm_results_ave, '-k',
                     label=f"FRCM", linestyle="-", linewidth=2.5, color="tab:purple")
        plt.fill_between(nunber_of_features, frcm_results_ave - frcm_results_std, frcm_results_ave + frcm_results_std,
                         color="tab:purple", alpha=0.4)

        plt.plot(nunber_of_features, frsd_results_ave, '-k',
                     label=f"FRSD", linestyle="-", linewidth=1.75, color="tab:red")
        plt.fill_between(nunber_of_features, frsd_results_ave - frsd_results_std, frsd_results_ave + frsd_results_std,
                         color="tab:red", alpha=0.4)

        plt.plot(nunber_of_features, frmv_results_ave,
                     label=f"FRMV", linestyle="-", linewidth=1, color="tab:green")
        plt.fill_between(nunber_of_features, frmv_results_ave - frmv_results_std, frmv_results_ave + frmv_results_std,
                         color="tab:green", alpha=0.4)


    plt.xticks(
        [1] + list(range(250, max(nunber_of_features.astype(int)) + 1, 250)) + [(nunber_of_features.astype(int)[-1])])
    plt.legend()
    plt.savefig(f"{path}{name}")
    plt.close()

    # Save the data the makes the plots
    all_data = pd.DataFrame(columns=nunber_of_features)
    all_data.loc["FRIGATE_ave"] = shacle_no_MW_ave
    all_data.loc["FRIGATE_std"] = shacle_no_MW_std
    all_data.loc["FRIGATE_MW_ave"] = shacle_05_ave
    all_data.loc["FRIGATE_MW_std"] = shacle_05_std


    if not include_cat:

        all_data.loc["FRMV_ave"] = frmv_results_ave
        all_data.loc["FRMV_std"] = frmv_results_std
        all_data.loc["FRSD_ave"] = frsd_results_ave
        all_data.loc["FRSD_std"] = frsd_results_std
        all_data.loc["FRCM_ave"] = frcm_results_ave
        all_data.loc["FRCM_std"] = frcm_results_std

    all_data.to_csv(f"{path}{name}.csv")

