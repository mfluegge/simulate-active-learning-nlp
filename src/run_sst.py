from strategies import USELogisticRegressionRandomStrategy
from strategies import USELogisticRegressionLeastConfidentStrategy#
from strategies import USELogisticRegressionEnsembleDisagreementStrategy
from strategies import USELogisticRegressionInformationDensityStrategy
from strategies import USELogisticRegressionLeastConfidentDiverseKMeansStrategy
from strategies import SetFitRandomStrategy
from experiments import run_experiment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    sst_data = pd.read_csv("../datasets/sst_train.tsv", sep="\t")

    texts = sst_data["sentence"].values[:20000]
    labels = sst_data["label"].values[:20000]

    texts, eval_x, labels, eval_y = train_test_split(
        texts, labels, test_size=.2, random_state=42
    )
    """

    strategy = USELogisticRegressionRandomStrategy(
        texts, label_per_iteration=20
    )

    run_experiment(
        strategy,
        texts,
        labels,
        eval_x,
        eval_y,
        output_path="../results/sst_use_random_20.json",
        experiment_name="USE+LR Random 20",
        dataset_name="SST",
        max_labeled_data=4000
    )
    strategy = USELogisticRegressionLeastConfidentStrategy(
        texts, label_per_iteration=1
    )

    run_experiment(
        strategy,
        texts,
        labels,
        eval_x,
        eval_y,
        output_path="../results/sst_use_least_confident_1.json",
        experiment_name="USE+LR Least Confident 1",
        dataset_name="SST",
        max_labeled_data=1500
    )

    strategy = USELogisticRegressionLeastConfidentStrategy(
        texts, label_per_iteration=10
    )

    run_experiment(
        strategy,
        texts,
        labels,
        eval_x,
        eval_y,
        output_path="../results/sst_use_least_confident_10.json",
        experiment_name="USE+LR Least Confident 10",
        dataset_name="SST",
        max_labeled_data=4000
    )

    strategy = USELogisticRegressionLeastConfidentStrategy(
        texts, label_per_iteration=40
    )

    run_experiment(
        strategy,
        texts,
        labels,
        eval_x,
        eval_y,
        output_path="../results/sst_use_least_confident_40.json",
        experiment_name="USE+LR Least Confident 40",
        dataset_name="SST",
        max_labeled_data=4000
    )



    strategy = USELogisticRegressionEnsembleDisagreementStrategy(
        texts, label_per_iteration=20
    )

    run_experiment(
        strategy,
        texts,
        labels,
        eval_x,
        eval_y,
        output_path="../results/sst_use_ensemble_diff_20.json",
        experiment_name="USE+LR Ensemble Diff 20",
        dataset_name="SST",
        max_labeled_data=4000
    )


    strategy = USELogisticRegressionInformationDensityStrategy(
        texts, label_per_iteration=20
    )

    run_experiment(
        strategy,
        texts,
        labels,
        eval_x,
        eval_y,
        output_path="../results/sst_use_information_density_20_05.json",
        experiment_name="USE+LR Information Density 20 beta=0.5",
        dataset_name="SST",
        max_labeled_data=4000
    )


    strategy = USELogisticRegressionLeastConfidentDiverseKMeansStrategy(
        texts, label_per_iteration=20
    )

    run_experiment(
        strategy,
        texts,
        labels,
        eval_x,
        eval_y,
        output_path="../results/sst_use_least_conf_kmeans_20_beta=10.json",
        experiment_name="USE+LR KMeans + Least Confident 20 beta=10",
        dataset_name="SST",
        max_labeled_data=4000
    )
    """
    strategy = SetFitRandomStrategy(
        texts, label_per_iteration=20
    )

    run_experiment(
        strategy,
        texts,
        labels,
        eval_x,
        eval_y,
        output_path="../results/sst_setfit_big_20_R=20_low_lr.json",
        experiment_name="SetFit Small 20 R=20 LR=2e-5",
        dataset_name="SST",
        max_labeled_data=800
    )