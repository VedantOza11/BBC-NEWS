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
The use of Bidirectional LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models for BBC News Classification proves beneficial. These models are prestigious for their viability in taking care of consecutive information, making them appropriate for normal language handling undertakings. The engineering includes layers of cells that cycle input successions bidirectionally, catching relevant data from both past and future data of interest.

The contribution to these models includes preprocessed message information, with tokenization, inserting, and cushioning utilized to mathematically address words. The model engineering normally comprises an implanting layer, trailed by Bidirectional LSTM or GRU layers. Extra layers, for example, dropout layers, are added for regularization, and a last thick layer with softmax enactment produces class expectations.

Preparing includes changing model loads in light of expectation mistakes to limit the misfortune capability. Hyperparameters like learning rate, dropout rate, and the quantity of LSTM or GRU units are tuned for ideal execution. The dataset is parted into preparing and testing sets, with normal grouping measurements like exactness, accuracy, review, and F1 score utilized for assessment.

Regularization methods, like dropout, forestall overfitting, guaranteeing the model sums up well to new information. Model interpretability is vital, and representations or consideration components might be utilized. Tweaking is iterative, with nonstop observing for ideal outcomes.

By coordinating Bidirectional LSTM and GRU models into the news characterization task, their arrangement handling capacities improve the model's capacity to catch unpredictable examples in the text information. This approach works with the exact order of BBC news stories, making it a powerful answer for robotized news characterization.

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


Explanation video: https://github.com/VedantOza11/BBC-NEWS/assets/114096362/8b8e0d64-741d-4b23-ba8c-935e570f62dc

