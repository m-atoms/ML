# ML Notes

## Machine Learning
* **definition**: A computer program is said to learn from experience *E* with respect to some task *T* and some performance measure *P*, if its performance on *T*, as measured by *P*, improves with exerpience *E* - Tom Mitchell
* **summary**: The biggest differentiator between machine learning and traditional programming is that with machine learning, the program is able to improve its own performance given sufficient and appropriate data 

## Training Methods: human involvement
### Supervised Learning
* **definition**: The training set given to the algorithm is labeled with the solution to the task
* **example**: A spam filter is given a data set of emails which have a label attached indicating wether they are spam or ham
* **common algorithms**
  * k-Nearest Neighbors
  * Linear Regression
  * Logistic Regression
  * Support Vector Machines (SVM)
  * Decision Trees & Random Forests
  * Neural Networks

### Unsupervised Learning
* **definition**: The training set given to the algorithm is unlabeled, the model is intended to uncover the structure/distribution of the input data on it own
* **example**: Discover clusters of users visiting a website, for instance when they visit, from where, for how long
* **common algorithms**
  * Clustering
    * K-Means
    * DBSCAN
    * Hierarchical Cluser Analysis (HCA)
  * Anomaly Detection & Novelty Detection
    * One-class SVM
    * Isolation Forest
  * Visualization and Dimensionality Reduction
    * Principal Component Analysis (PCA)
    * Kernel PCA
    * Locally Linear Embedding (LLE)
    * t-Distributed Stochastic Neighbor Embedding (t-SNE)
  * Association Rule Learning
    * Apriori
    * Eclat

### Semisupervised Learning
* **definition**: The data set given to the algorithm will have mostly unlabeled data with a portion of labeled data included. Most semisupervised learning algorithims are combinations of supervised and unsupervised.
* **example**: Photos app clusters groups into "people" then user adds the label (name), then future pictures of those people can be labeled

### Reinforcement Learning
* **definition**: The algorithm (agent) observes the enviornment, chooses which actions to perform, and gets a reward/penalty based on the outcome of that action. The resulting strategy is called the policy.
* **example**: The AlphaGo algorithm analyzed millions of games and learned a policy sophisticated enough to beat the world champion human

## Training Methods: operational status
### Batch Learning
* **definition**: Learning does not occur incrementally. Algorithm is trained on entire set of data before being launched and does not learn in production environment

### Online Learning
* **definition**: Learning occurs incrementally by feeding new data instances in sequentially or in small batches.

## Prediction Methods
### Instance-Based
* **definition**: the algorithim learns the examples and then generalizes to new cases using measures of similarity
* **example**: spam filter detects that an email it was never trained on is spam based on similarity of particular attributes (like word frequency) to previously learned spam email examples

### Model-Based
* **definition**: build a model from training data and then use the model to make predictions
* **example**: establish that country GDP and citizen happiness are linearly related, then use resulting model to predict happiness in other countries based on their GDP

## Main Challenges of Machine Learning
### Bad Data
* **Insufficient Quantity of Training Data**: 
* **Nonrepresentative Training Data**:
* **Poor-Quality Data**:
### Bad Algorithm
* **Irrelevant Features**:
* **Overfitting Training Data**:
* **Underfitting Training Data**:
