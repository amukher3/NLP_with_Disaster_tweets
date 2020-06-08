# NLP_with_Disaster_tweets
Kaggle_NLP_challenge

This is an effort to categorize and classify `Real` disaster tweets from `Fake` ones. The data consists of location as one of the features and the text as another feature. I have just used the text or the tweet for the classification problem.I have used a regularized LSTM as the model for the sequence classification task.A 2-layered LSTM is used after the Embedding layer.The model is regularised carefully using drop-out(regular as well as spatial). The model achieved a `validation accuracy of 0.80` on a decent sized test-set.

Using pre-trained models such as BERT,ALBERTA,ELECTRA etc might end up giving a better accuracy. I intend to try this in future after making some changes. 

In a separate notebook,I have further explored the data-set by taking the location into account and exploring the top-10 most frequently mentinoned location for each category. 

***Author:Abhishek Mukherjee***


