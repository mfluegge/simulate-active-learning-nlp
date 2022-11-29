import matplotlib.pyplot as plt
import json
import numpy as np


if __name__ == "__main__":
    results_paths = [
        "../results/sst_use_random_20.json",
        #"../results/sst_use_least_confident_1.json",
        #"../results/sst_use_least_confident_10.json",
        "../results/sst_use_least_confident_20.json",
        #"../results/sst_use_least_confident_40.json",
        #"../results/sst_use_ensemble_diff_20.json",
        "../results/sst_use_information_density_20.json",
        #"../results/sst_use_information_density_20_05.json",
        "../results/sst_use_least_conf_kmeans_20_beta=10.json",
        #"../results/sst_setfit_small_20_R=10.json",
        "../results/sst_setfit_small_20_R=10_low_lr.json",
        "../results/sst_setfit_big_20_R=20_low_lr.json"
    ]

    time_per_label = 3

    plt.figure(figsize=(12, 10))
    for path in results_paths:
        with open(path, "r") as f:
            data = json.load(f)

        x_labeled = []
        x_time = []
        y_acc = []
        init_time = data["pre_time"]
        for step in data["labeling_steps"]:
            x_labeled.append(step["num_labeled"])
            y_acc.append(step["accuracy"])
            x_time.append(step["time_taken"] + time_per_label * step["num_labeled"])

        x_labeled = np.cumsum(x_labeled)
        x_time = np.cumsum(x_time) + init_time

        plt.plot(x_labeled, y_acc, label=data["experiment_name"])
        #plt.plot(x_time, y_acc, label=data["experiment_name"])

    plt.xlabel("Number of labeled examples")
    plt.title(f"Accuracy on dataset {data['dataset_name']} in relation to time spent (assuming 3 seconds per label)")
    #plt.xlabel("Time taken (seconds)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

