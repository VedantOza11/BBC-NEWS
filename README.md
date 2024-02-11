# ThinkML Competition

Team no: 10


Dataset: News Classification

Problem statement: Build a machine learning model to automatically categorize BBC news articles into the correct category (business, entertainment, politics, sport, or tech).


Team Members,

Name: Suraj Mishra,
Department: T.E(E&CS),
Roll no: 31,
Email Id: mishrasuraj2853@gmail.com,
Mob: 8830303424

Name: Vedant Oza,
Department: T.E(E&CS),
Roll no: 35,
Email Id: vedantoza0@gmail.com,
Mob: 9967529130

Name: Prathamesh Panchal,
Department: T.E(E&CS),
Roll no: 36,
Email Id: prathameshpanchal302@gmail.com,
Mob: 8879316175

RapidMiner :[ThinkML_RapidMiner_Team10(News Classification).pdf](https://github.com/VedantOza11/BBC-NEWS/files/14232295/ThinkML_RapidMiner_Team10.News.Classification.pdf)


INTRODUCTION:
Introducing our new news classification model, which is the result of intensive study and testing with the goal of maximizing efficiency and accuracy. Our model combines the advantages of both approaches—Bidirectional Long Short-Term Memory (BiLSTM) and Gated Recurrent Unit (GRU) algorithms—to capture complex contextual connections found in BBC news pieces. Even though our hybrid architecture uses both GRU and BiLSTM, empirical results show that the model that uses solely the output of Bidirectional LSTM consistently achieves the maximum accuracy. By automating the classification process into discrete categories like business, entertainment, politics, sport, or technology, the suggested news classification model tackles the massive amount of digital news material. By effectively sorting through articles, offering individualized content recommendations, and enabling quick access to pertinent news, this model saves time in our daily lives.

DATASET DESCRIPTION:
The dataset comprises three separate files: a training dataset, a test dataset, and a sample solution dataset for BBC news classification.

Training Dataset:
Rows: 3,
Columns: 3

Contents: Each row represents an article with three columns: 'article id', 'text', and 'category'. The 'article id' uniquely identifies each article, 'text' contains the textual content of the article, and 'category' specifies the category to which the article belongs (e.g., business, entertainment, politics, sport, or technology).

Test Dataset:
Rows: Varies,
Columns: 2

Contents: Each row represents an article with two columns: 'article id' and 'text'. The 'article id' serves as a unique identifier for each article, while 'text' contains the textual content of the news article. Unlike the training dataset, the 'category' column is absent, as it's not provided in the test dataset.

Sample Solution Dataset:
Rows: 2,
Columns: 3

Contents: This dataset provides a sample solution for the test dataset. Each row corresponds to an article in the test dataset and contains the 'article id', 'text', and the predicted 'category' column. The predicted category indicates the category predicted by the model for each article.
These datasets are structured to facilitate the training, evaluation, and testing of machine learning models for news categorization tasks. The training dataset is used to train the model, while the test dataset evaluates the model's performance on unseen data. The sample solution dataset offers a reference for comparing model predictions against ground truth labels.


DATASET SPLIT:
1. Training Set (70-80%):
   - Larger part of the dataset allotted for preparing the Bidirectional LSTM and GRU models.
   - Utilized for learning complex examples and connections between printed elements and classes.
   - Makes it possible for the models to apply to other training data.

2. Validation Set (10-15%):
   - A smaller subset used for tuning hyperparameters and selecting models during training.
   - Fundamental for forestalling overfitting and guaranteeing ideal model execution.
   - Execution on the approval set guides changes in accordance with the model design and hyperparameters.

3. Testing Set (10-20%):
   - Remaining piece of the dataset held totally different during model preparation.
   - Serves as an unseen benchmark for the evaluation of the final model.
   - Evaluates the model's speculation to new, true information.

The dataset split into preparing, approval, and testing sets is a urgent part of building a hearty AI model for BBC News Order. The preparation set gives the fundamental information to the models, permitting them to gain from a different scope of models. The approval set supports tweaking, guaranteeing the model's boundaries are advanced without overfitting to the preparation information. The testing set fills in as a definitive benchmark, surveying the model's presentation on totally original cases and giving experiences into its genuine relevance. This precise methodology guarantees that the model is prepared successfully, refined fittingly, and assessed thoroughly, adding to its dependability in arranging BBC news stories across particular topical classes.

APPROACH:
In our project blueprint, we initially employed the KMeans algorithm in RapidMiner due to the unavailability of LSTM and GRU models. However, we encountered limitations with the KMeans output, which did not align with the requirements of our news classification project.

To overcome these challenges and enhance our model's capabilities, we shifted to Python, where we leveraged Bidirectional LSTM and GRU architectures. This strategic decision aims to mitigate potential drawbacks associated with using a single model type. By combining both Bidirectional LSTM and GRU models, we can benefit from their complementary strengths, ensuring a more robust and accurate classification system for BBC news articles.

This dual-model approach allows us to capture bidirectional sequential dependencies efficiently, enhancing the model's ability to understand and categorize diverse textual information. The utilization of LSTM and GRU in Python reflects our commitment to achieving the best possible performance and overcoming the limitations identified during the initial stages of the project in RapidMiner.


RESEULT:![Output](https://github.com/VedantOza11/BBC-NEWS/assets/114096362/ced2571f-82ab-4ac7-8272-a5d42c3a5f55)

DEPENDENCIES: Dependencies for a project involves building a machine learning model for news classification using Bidirectional LSTM and GRU algorithms, can vary depending on the specific tools and frameworks chosen for development. Here's a list of potential dependencies:

Python: The primary programming language for implementing machine learning models.

Deep Learning Frameworks: Libraries such as TensorFlow, PyTorch, or Keras provide tools for building and training neural networks, including Bidirectional LSTM and GRU layers.

NLP Libraries: Libraries like NLTK (Natural Language Toolkit) or SpaCy can be used for natural language processing tasks such as tokenization, stemming, and lemmatization.

Data Processing Libraries: Pandas and NumPy are commonly used for data manipulation and preprocessing tasks.

Text Vectorization Techniques: Tools for converting text data into numerical vectors, such as TF-IDF Vectorizer or Word Embeddings (Word2Vec, GloVe).

Model Evaluation Metrics: Libraries like Scikit-learn provide functions for evaluating model performance using metrics such as accuracy, precision, recall, and F1-score.

Development Environment: Tools such as Jupyter Notebook, Google Colab, or IDEs like PyCharm or VSCode provide environments for code development, experimentation, and debugging.

Version Control: Platforms like Git and hosting services like GitHub or GitLab facilitate collaboration, version control, and code sharing among team members.

Documentation and Reporting: Tools like Jupyter Notebook, Markdown, or LaTeX can be used for documenting project progress, findings, and results.

Deployment and Integration: Depending on the project requirements, additional dependencies may include frameworks for deploying models to production environments (e.g., Flask, Django, or TensorFlow Serving) and integrating them with other systems or applications.

These dependencies provide the necessary tools and resources for developing, training, evaluating, and deploying machine learning models for news classification effectively. It's essential to carefully select and manage these dependencies to ensure compatibility, scalability, and maintainability throughout the project lifecycle.

PERFORMANCE AND ACCURACY:

Final Training Accuracy: 0.999104

Final Validation Accuracy:0.914209

Final Training Loss:0.010594

NOVELTY FACTOR:

Integrating Bidirectional Long Short-Term Memory (BiLSTM) and Gated Recurrent Unit (GRU) algorithms for news classification represents a novel approach. This combination harnesses bidirectional processing, enhancing the model's capability to capture contextual dependencies in both forward and backward directions within news articles.

By leveraging both Bidirectional LSTM and GRU networks, our model demonstrates improved performance in understanding the sequential nature of text data. This hybrid architecture is particularly beneficial for tasks like news categorization, where contextual understanding is crucial for accurate classification.

Notably, our experimentation revealed that while both the combined output of LSTM and GRU and the model solely utilizing Bidirectional LSTM yielded promising results, the latter consistently provided the best output. This observation underscores the effectiveness of Bidirectional LSTM in news classification tasks, highlighting its superiority in capturing relevant contextual information for accurate categorization.


EXPLANATION VIDEO: https://github.com/VedantOza11/BBC-NEWS/assets/114096362/8b8e0d64-741d-4b23-ba8c-935e570f62dc

References: https://www.kaggle.com/c/learn-ai-bbc
