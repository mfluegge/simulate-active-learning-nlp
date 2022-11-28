import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

#LOGGER = get_logger(__name__)

class LabelingStrategy:
    def __init__(self, texts, label_per_iteration=10):
        self.texts = texts
        self.label_per_iteration = label_per_iteration

    def get_init_examples(self):
        random_indices = np.random.choice(np.arange(len(self.texts)), size=self.label_per_iteration, replace=False)
        return random_indices
    
    def setup_labeling(self):
        pass
    
    def setup_prediction(self):
        pass
    
    def get_next_examples(self):
        raise NotImplementedError
    
    def add_labels(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError
    
    def post_labeling_step(self):
        pass
    
    def post_labeling_predict(self, eval_texts):
        return self.predict(eval_texts)


class USELogisticRegressionRandomStrategy(LabelingStrategy):
    def __init__(self, texts, label_per_iteration=10, use=None):
        super().__init__(texts, label_per_iteration)

        if use is None:
            logging.info("Loading USE")
            self.use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        else:
            self.use = use

        self.labeled_indices = []
        self.labels = []
        self.latest_clf = None
        
        self.text_embeddings = None
        self.eval_text_embeddings = None

    def setup_labeling(self):
        logging.debug("Embedding texts for USE random strategy")
        self.text_embeddings = self.use(self.texts).numpy()

    def setup_prediction(self, eval_texts):
        logging.debug("Embedding texts for USE random strategy (prediction only)")
        self.eval_text_embeddings = self.use(eval_texts).numpy()

    def get_next_examples(self):
        unlabeled_indices = np.setdiff1d(
            np.arange(len(self.texts)), self.labeled_indices
        )

        if len(unlabeled_indices) <= self.label_per_iteration:
            return unlabeled_indices

        next_indices = np.random.choice(
            unlabeled_indices,
            size=self.label_per_iteration, 
            replace=False
        )

        return next_indices
    
    def add_labels(self, indices, labels):
        self.labeled_indices += indices
        self.labels += labels
    
    def predict(self, _):
        logging.debug("Training model with labeled examples for prediction")
        clf = LogisticRegression(random_state=42).fit(self.text_embeddings[self.labeled_indices], self.labels)

        logging.debug("Predicting validation texts")
        pred = clf.predict(self.eval_text_embeddings)

        return pred


class USELogisticRegressionLeastConfidentStrategy(LabelingStrategy):
    def __init__(self, texts, label_per_iteration=10, use=None):
        super().__init__(texts, label_per_iteration)

        if use is None:
            logging.info("Loading USE")
            self.use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        else:
            self.use = use

        self.labeled_indices = []
        self.labels = []
        self.latest_clf = None
        
        self.text_embeddings = None
        self.eval_text_embeddings = None

    def get_init_examples(self):
        random_indices = np.random.choice(np.arange(len(self.texts)), size=10, replace=False)
        return random_indices
    

    def setup_labeling(self):
        logging.debug("Embedding texts for USE Least Confident strategy")
        self.text_embeddings = self.use(self.texts).numpy()

    def setup_prediction(self, eval_texts):
        logging.debug("Embedding texts for USE random strategy (prediction only)")
        self.eval_text_embeddings = self.use(eval_texts).numpy()

    def get_next_examples(self):
        unlabeled_indices = np.setdiff1d(
            np.arange(len(self.texts)), self.labeled_indices
        )

        if len(unlabeled_indices) <= self.label_per_iteration:
            return unlabeled_indices


        max_conf = self.latest_clf.predict_proba(self.text_embeddings[unlabeled_indices]).max(axis=1)
        next_indices = unlabeled_indices[np.argsort(max_conf)[:self.label_per_iteration]]

        return next_indices
    
    def add_labels(self, indices, labels):
        self.labeled_indices += indices
        self.labels += labels

        self.latest_clf = LogisticRegression(random_state=42).fit(self.text_embeddings[self.labeled_indices], self.labels)

    def predict(self, _):
        logging.debug("Predicting validation texts")
        pred = self.latest_clf.predict(self.eval_text_embeddings)

        return pred


class USELogisticRegressionEnsembleDisagreementStrategy(LabelingStrategy):
    def __init__(self, texts, label_per_iteration=10, use=None):
        super().__init__(texts, label_per_iteration)

        if use is None:
            logging.info("Loading USE")
            self.use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        else:
            self.use = use

        self.labeled_indices = []
        self.labels = []
        self.latest_ensemble = []
        self.latests_clf = None
        
        self.text_embeddings = None
        self.eval_text_embeddings = None


    def setup_labeling(self):
        logging.debug("Embedding texts for USE Least Confident strategy")
        self.text_embeddings = self.use(self.texts).numpy()

    def setup_prediction(self, eval_texts):
        logging.debug("Embedding texts for USE random strategy (prediction only)")
        self.eval_text_embeddings = self.use(eval_texts).numpy()

    def get_next_examples(self):
        unlabeled_indices = np.setdiff1d(
            np.arange(len(self.texts)), self.labeled_indices
        )

        if len(unlabeled_indices) <= self.label_per_iteration:
            return unlabeled_indices

        first_clf_preds = self.latest_ensemble[0].predict_proba(self.text_embeddings[unlabeled_indices])[:, 1]
        second_clf_preds = self.latest_ensemble[1].predict_proba(self.text_embeddings[unlabeled_indices])[:, 1]
        diffs = abs(first_clf_preds - second_clf_preds)
        next_indices = unlabeled_indices[np.argsort(diffs * -1)[:self.label_per_iteration]]

        return next_indices
    
    def add_labels(self, indices, labels):
        self.labeled_indices += indices
        self.labels += labels

        first_clf_indices, second_clf_indices, first_clf_labels, second_clf_labels = train_test_split(
            self.labeled_indices, self.labels, test_size=0.5, random_state=42
        )


        first_clf = LogisticRegression(random_state=42).fit(self.text_embeddings[first_clf_indices], first_clf_labels)
        second_clf = LogisticRegression(random_state=42).fit(self.text_embeddings[second_clf_indices], second_clf_labels)

        self.latest_clf = LogisticRegression(random_state=42).fit(self.text_embeddings[self.labeled_indices], self.labels)
        self.latest_ensemble = [first_clf, second_clf]

    def predict(self, _):
        logging.debug("Predicting validation texts")
        pred = self.latest_clf.predict(self.eval_text_embeddings)

        return pred


class USELogisticRegressionInformationDensityStrategy(LabelingStrategy):
    def __init__(self, texts, label_per_iteration=10, use=None):
        super().__init__(texts, label_per_iteration)

        if use is None:
            logging.info("Loading USE")
            self.use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        else:
            self.use = use

        self.labeled_indices = []
        self.labels = []
        self.latest_clf = None
        
        self.text_embeddings = None
        self.eval_text_embeddings = None

        self.text_to_text_sims = None

    def setup_labeling(self):
        logging.debug("Embedding texts for USE Least Confident strategy")
        self.text_embeddings = self.use(self.texts).numpy()

        self.text_to_text_sims = cosine_similarity(self.text_embeddings)


    def setup_prediction(self, eval_texts):
        logging.debug("Embedding texts for USE random strategy (prediction only)")
        self.eval_text_embeddings = self.use(eval_texts).numpy()

    def get_next_examples(self):
        unlabeled_indices = np.setdiff1d(
            np.arange(len(self.texts)), self.labeled_indices
        )

        if len(unlabeled_indices) <= self.label_per_iteration:
            return unlabeled_indices

        max_conf = 1 - self.latest_clf.predict_proba(self.text_embeddings[unlabeled_indices]).max(axis=1)


        text_2_text_matrix = self.text_to_text_sims[unlabeled_indices]
        text_2_text_matrix = text_2_text_matrix[:, unlabeled_indices]

        mean_sim = np.mean(text_2_text_matrix, axis=1) ** 0.5

        score = (max_conf * mean_sim) * -1
        next_indices = unlabeled_indices[np.argsort(score)[:self.label_per_iteration]]

        return next_indices
    
    def add_labels(self, indices, labels):
        self.labeled_indices += indices
        self.labels += labels

        self.latest_clf = LogisticRegression(random_state=42).fit(self.text_embeddings[self.labeled_indices], self.labels)

    def predict(self, _):
        logging.debug("Predicting validation texts")
        pred = self.latest_clf.predict(self.eval_text_embeddings)

        return pred



class USELogisticRegressionLeastConfidentDiverseKMeansStrategy(LabelingStrategy):
    def __init__(self, texts, label_per_iteration=10, use=None, beta=10):
        super().__init__(texts, label_per_iteration)

        if use is None:
            logging.info("Loading USE")
            self.use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        else:
            self.use = use

        self.labeled_indices = []
        self.labels = []
        self.latest_clf = None
        
        self.text_embeddings = None
        self.eval_text_embeddings = None
        self.beta = beta

    def get_init_examples(self):
        random_indices = np.random.choice(np.arange(len(self.texts)), size=10, replace=False)
        return random_indices

    def setup_labeling(self):
        logging.debug("Embedding texts for USE Least Confident strategy")
        self.text_embeddings = self.use(self.texts).numpy()

    def setup_prediction(self, eval_texts):
        logging.debug("Embedding texts for USE random strategy (prediction only)")
        self.eval_text_embeddings = self.use(eval_texts).numpy()

    def get_next_examples(self):
        unlabeled_indices = np.setdiff1d(
            np.arange(len(self.texts)), self.labeled_indices
        )

        if len(unlabeled_indices) <= self.label_per_iteration:
            return unlabeled_indices


        max_conf = self.latest_clf.predict_proba(self.text_embeddings[unlabeled_indices]).max(axis=1)

        best_uncertainty_indices = unlabeled_indices[np.argsort(max_conf)[:self.label_per_iteration * self.beta]]
        best_uncertainty_embeds = self.text_embeddings[best_uncertainty_indices]

        km = KMeans(n_clusters=self.label_per_iteration, random_state=42).fit(best_uncertainty_embeds)
        
        next_indices = []
        for cluster in np.unique(km.labels_):
            cluster_embs = best_uncertainty_embeds[km.labels_ == cluster]
            cluster_indices = best_uncertainty_indices[km.labels_ == cluster]
            cluster_center_sims = cosine_similarity(cluster_embs, [km.cluster_centers_[cluster]])[:, 0]
            max_sim_ix = np.argmax(cluster_center_sims)
            best_cluster_ix = cluster_indices[max_sim_ix]
            next_indices.append(best_cluster_ix)

        return np.array(next_indices)
    
    def add_labels(self, indices, labels):
        self.labeled_indices += indices
        self.labels += labels

        self.latest_clf = LogisticRegression(random_state=42).fit(self.text_embeddings[self.labeled_indices], self.labels)

    def predict(self, _):
        logging.debug("Predicting validation texts")
        pred = self.latest_clf.predict(self.eval_text_embeddings)

        return pred
