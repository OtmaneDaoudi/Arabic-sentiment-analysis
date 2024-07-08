This branch contains the code for sentiment analysis using the BERT model.

# Performance evaluation

| model | with emojis| training dataset  |  arabic-sentiment-twitter-corpus accuracy | SS2030 accuracy | 100k reviews accuracy  
| :---: | :---: | :---: |  :---: | :---: | :---:
| RandomForestClassifier | Yes | arabic-sentiment-twitter-corpus | *0.798*| 0.550 | 0.585
| BERT-mini| Yes | arabic-sentiment-twitter-corpus | 0.900| 0.639 | 0.599
| BERT-mini| No | arabic-sentiment-twitter-corpus | 0.785| 0.579 | 0.570
| BERT-base | No | arabic-sentiment-twitter-corpus | 0.803  |  0.652| 0.652
| BERT-mini| No | SS2030  | 0.492 | 0.847| 0.502  
| BERT-mini| No | 100k reviews  | 0.585 | 0.626 | *0.885*
