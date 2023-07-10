# FRIGATE - Feature Ranking In clustering using GAme ThEory
FRIGATE is a feature ranking algorithm for clustering, designed for medical data. Details of the algorithm are found in the thesis work of Shpigelman Eran from 2023 at Tel Aviv University, under the title “A Feature Ranking Algorithm for Clustering Medical Data”.  
This project consists of the following files: 
1.	README – This file. 
2.	FRIGATE.py – Code of the FRIGATE algorithm.
3.	FRSD.py - Code of the FRSD algorithm, an ensemble feature ranking algorithm, first described by Yu et al., 2020.
4.	FRMV.py - Code of the FRMV algorithm, an ensemble feature ranking algorithm, first described by Hong et al., 2008.
5.	FRCM.py - Code of the FRCM algorithm, an ensemble feature ranking algorithm, first described by Zhang et al., 2012.
6.	compare_genomic_data.py – the code for comparing the performance of FRIGATE, FRSD, FRMV, FRCM on a series of cancer genomic data, used in a clustering benchmark by Souto et al., 2008. The Datasets are available at: https://schlieplab.org/Static/Supplements/CompCancer/datasets.htm
7.	simulation.py – the code for generating a simulation and comparing the performance of FRIGATE, FRSD, FRMV, FRCM, as described by Shpigelman et al. 2023.
8.	compare_pipeline.py – the code executed by compare_genomic_data.py and simulation.py that execute the algorithms and compare their performances.
 
