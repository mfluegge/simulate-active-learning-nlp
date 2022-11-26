import logging
import time
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

def run_experiment(
    strategy,
    texts,
    labels,
    eval_x,
    eval_y,
    output_path,
    max_labeled_data=None,
    experiment_name="too lazy to think of a name",
    dataset_name="too lazy to think of dataset name"
):
    logging.info(f"Starting experiment {experiment_name}")
    if max_labeled_data is None:
        max_labeled_data = len(texts)
    
    logs = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "pre_time": None,
        "labeling_steps": [],
        "post_time": None,
        "final_accuracy": None
    }
    total_labeled_examples = 0
    total_time_taken = 0

    logging.info("Starting to measure time")
    start_time = time.time()

    logging.info("Running labeling preparation")
    strategy.setup_labeling()
    setup_labeling_time = time.time() - start_time
    logs["pre_time"] = setup_labeling_time

    logging.info("Running prediction preparation")
    strategy.setup_prediction(eval_x)

    logging.info("Starting simulation")
    while total_labeled_examples < min(len(texts), max_labeled_data):
        iteration_start = time.time()
        if total_labeled_examples == 0:
            logging.debug("Getting initial examples")
            label_indices = strategy.get_init_examples()
        else:
            label_indices = strategy.get_next_examples()
        
        # Simulating the labeling process
        example_labels = labels[label_indices]

        strategy.add_labels(label_indices.tolist(), example_labels.tolist())

        iteration_time = time.time() - iteration_start
        
        iteration_preds = strategy.predict(eval_x)
        iteration_acc = accuracy_score(eval_y, iteration_preds)

        logs["labeling_steps"].append({
            "num_labeled": len(label_indices),
            "time_taken": iteration_time,
            "accuracy": iteration_acac
        })

        iteration_labeled_examples += len(label_indices)
    
    logging.info("Done with labeling, running post_labeling process")
    start_time = time.time()
    strategy.post_labeling_step()
    post_labeling_time = time.time() - start_time
    logs["post_time"] = post_labeling_time

    logging.info("Running final prediction")
    final_pred = strategy.post_labeling_predict(eval_texts)
    final_acc = accuracy_score(eval_y, final_pred)

    logs["final_accuracy"] = final_acc

    logging.info(f"Saving results at {output_path}")
    with open(output_path, "w") as f:
        json.dump(logs, f, indent=4)

    logging.info("All done")
    






