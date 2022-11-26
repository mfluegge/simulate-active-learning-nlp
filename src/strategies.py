import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import logging
from sklearn.linear_model import LogisticRegression

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
        clf = LogisticRegression().fit(self.text_embeddings[labeled_indices]. self.labels)

        logging.debug("Predicting validation texts")
        pred = clf.predict(self.eval_text_embeddings)

        return pred

