# BBC-NEWS
ThinkML Competition

Team no: 10

Dataset: News Classification

Problem statement: Build a machine learning model to automatically categorize BBC news articles into the correct category (business, entertainment, politics, sport, or tech).

Team Members,

Name: Suraj Mishra
Department: T.E(E&CS)
Roll no: 31
Email Id: mishrasuraj2853@gmail.com
Mob: 8830303424

Name: Vedant Oza
Department: T.E(E&CS)
Roll no: 35
Email Id: vedantoza0@gmail.com
Mob: 9967529130

Name: Prathamesh Panchal 
Department: T.E(E&CS)
Roll no: 36
Email Id: prathameshpanchal302@gmail.com
Mob: 8879316175

RapidMiner :[ThinkML_RapidMiner_Team10(News Classification).pdf](https://github.com/VedantOza11/BBC-NEWS/files/14232295/ThinkML_RapidMiner_Team10.News.Classification.pdf)


Introduction:
Introducing our new news classification model, which is the result of intensive study and testing with the goal of maximizing efficiency and accuracy. Our model combines the advantages of both approaches—Bidirectional Long Short-Term Memory (BiLSTM) and Gated Recurrent Unit (GRU) algorithms—to capture complex contextual connections found in BBC news pieces. Even though our hybrid architecture uses both GRU and BiLSTM, empirical results show that the model that uses solely the output of Bidirectional LSTM consistently achieves the maximum accuracy. By automating the classification process into discrete categories like business, entertainment, politics, sport, or technology, the suggested news classification model tackles the massive amount of digital news material. By effectively sorting through articles, offering individualized content recommendations, and enabling quick access to pertinent news, this model saves time in our daily lives.

Dataset Description:
The use of Bidirectional LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models for BBC News Classification proves beneficial. These models are prestigious for their viability in taking care of consecutive information, making them appropriate for normal language handling undertakings. The engineering includes layers of cells that cycle input successions bidirectionally, catching relevant data from both past and future data of interest.

The contribution to these models includes preprocessed message information, with tokenization, inserting, and cushioning utilized to mathematically address words. The model engineering normally comprises an implanting layer, trailed by Bidirectional LSTM or GRU layers. Extra layers, for example, dropout layers, are added for regularization, and a last thick layer with softmax enactment produces class expectations.

Preparing includes changing model loads in light of expectation mistakes to limit the misfortune capability. Hyperparameters like learning rate, dropout rate, and the quantity of LSTM or GRU units are tuned for ideal execution. The dataset is parted into preparing and testing sets, with normal grouping measurements like exactness, accuracy, review, and F1 score utilized for assessment.

Regularization methods, like dropout, forestall overfitting, guaranteeing the model sums up well to new information. Model interpretability is vital, and representations or consideration components might be utilized. Tweaking is iterative, with nonstop observing for ideal outcomes.

By coordinating Bidirectional LSTM and GRU models into the news characterization task, their arrangement handling capacities improve the model's capacity to catch unpredictable examples in the text information. This approach works with the exact order of BBC news stories, making it a powerful answer for robotized news characterization.

Dataset Split Info:
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

Approach:
The approach for building a BBC News Classification model involves initial data preprocessing, including tokenization and exploratory data analysis (EDA).The dataset is then decisively parted into preparing, approval, and testing sets to empower powerful model preparation, hyperparameter tuning, and last assessment. The brain network design contains inserting layers followed by Bidirectional LSTM and GRU layers to catch bidirectional consecutive conditions, enhanced with extra layers like dropout for regularization. Hyperparameters are tweaked through strategies, for example, framework search.

During Testing, the model's presentation is checked on the approval set to forestall overfitting. The last assessment evaluates the model's exactness, accuracy, review, and F1 score on the testing set. Interpretability strategies, like consideration systems, give bits of knowledge into the model's dynamic cycle. If necessary, the model goes through tweaking in view of assessment experiences, and the iterative cycle go on until ideal execution is accomplished. A methodical and efficient machine learning development cycle is completed when the model is used to automatically classify BBC news articles into business, entertainment, politics, sport, and technology.

Results:
