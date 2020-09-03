# OcadoReviewCode
 An analysis code for ocado reivews which contain a sentiment and topic model based on word embedding.

 ## Code structure
 * data/ : The folder used to save original ocado data excel files. And data for ETM model will be generated here
 * model/ : The folder used to save CNN, ETM, LDA model files.
 * preprocess/ : The scripts which is used to generate data as models' inputs.
 * results/ : The folder used to save csv, png results.
 * checkpoints/ : The folder used to save trained models' parameters.
 * sentiment_model_train.py : The python file used to train sentiment model.
 * sentiment.py : The python file used to do sentiment classification
 * topic_model.py : The python file used to train or generate topics

 ## Guide
 
 ### Train Sentiment Model
 For the first step, we need to pretrain the sentiment models which based on CNN by a open source labeled dataset.
 ```shell
 python sentiment_model_train.py
 ```
 And it will save the trained model parameters in ```checkpoints/```.

 ### Sentiment Classify
 Load the trained sentiment data model, to preprocess and classify the Ocado raw parsed data.
 ```shell
 python sentiment.py
 ```
 It will generate a ```processed_sentence.csv``` in ```results/``` and it includes splited sentence, emotion score of each sentence and review date.
 The ```id``` is the unique identification of each sentence, and ```content_id``` shows that the sentence is belong to which review in original data.

 ### Preprocess data for Topic Model
 Before generating topics, data is needed to be preprocessed
 ```shell
 cd preprocess
 python ETM_data_process.py
 ```
 A splited training and validing data will be generated in ```data/```.
 Then,
 ```shell
 python skipgram.py
 ```
 It will generate word embeddings by word2vec model, and write the word embeddings into a ```.txt``` file which will be used in ETM.

 ### Topic Model
 Go to the root folder of this project and run:
 ```shell
 python topic_model.py --num_topics 5 --epochs 500
 ```
 It is for training , the argument ```--num_topics``` indicates how many topics will be produced, it's a hyperparameter.

 For evaluating, run:
 ```shell
 python topic_model.py --num_topics 5 --load_from xxx --mode eval
 ```
 The argument ```--num_topics``` should be same with trained model. ```--load_from``` indicates which model should be load, and all models will be saved in ```checkpoints/```

 ### Credits

 The model in this project is referenced repo as followed:

 ETM: https://github.com/adjidieng/ETM
