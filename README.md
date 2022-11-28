# simulate-active-learning-nlp
Simulate different Active Learning approaches for NLP tasks 


## Interesting papers
* http://charuaggarwal.net/active-survey.pdf
* https://arxiv.org/abs/1901.05954


## Ideas
* Move to Google Cloud Instance
* Try with SetFit
* If reducing average uncertainty for the unlabeled set is a good proxy for classifier performance
  * maybe one can run a normal strategy and collect data about how certain chosen queries reduced unlabeled uncertainty
  * train a model with data
  * Switch strategy at some point: Instead use model to score unlabeled examples, maximize uncertainty reduction