from strategies import USELogisticRegressionRandomStrategy
from experiments import run_experiment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    sst_data = pd.read_csv("../datasets/sst_train.tsv", sep="\t")

    texts = sst_data["sentence"].values
    labels = sst_data["label"].values

    texts, eval_x, labels, eval_y = train_test_split(
        texts, labels, test_size=.2, random_state=42
    )

    strategy = USELogisticRegressionRandomStrategy(
        texts, label_per_iteration=20
    )

    run_experiment(
        strategy,
        texts,
        labels,
        eval_x,
        eval_y,
        output_path="../datasets/sst_use_random_20.json",
        experiment_name="USE+LR Random 20",
        dataset_name="SST",
        max_labeled_data=2000
    )
