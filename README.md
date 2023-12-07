# Connectivity-aware Experience Replay for Graph Convolution Network-based Collaborative Filtering in Incremental Setting

Abstract—With the explosive growth of data on the Internet, recommender systems are playing an increasingly crucial role in various domains of the real world to alleviate the information overload problem and improve user experience. Over the past few years, the emergence of graph convolution networks (GCNs) has enhanced collaborative filtering-based recommender systems owing to their superiority in capturing higher-order relationships. However, GCNs are associated with a low training efficiency because they require multiple iterations for information aggregation, hindering their widespread application in large-scale recommender systems. In an incremental setting, because of this drawback, GCN-based models produce infrequent updates when facing successive data streams, resulting in the loss of short-term user preferences. Moreover, the dynamically changing graph structures and user preferences make it difficult for GCN-based models to preserve the long-term preferences of users. To solve this issue, we present a novel method based on experience replay to train GCN-based models in the incremental setting with short time intervals and high accuracy. Specifically, we propose a novel sampling mechanism named “connectivity sampling” customized for graph structures. Connectivity sampling is based on the graph connectivity of a newly appeared set of interactions to perform rich graph representation learning, thereby preserving the long- term user preferences while capturing the short-term ones. Comprehensive experimental evaluations on the Gowalla and Yelp datasets demonstrated the superior performance of our method, which significantly reduced the training time required to update the GCN, outperforming the state-of-the-art method “inverse degree,” as evidenced by improvements in the recall, NDCG, and precision metrics. The improvement in Recall@10 was 4.495% from 0.0861 to 0.0899 on the Gowalla dataset and 3.989% from 0.0461 to 0.0479 on the Yelp dataset.
<br>
<br>
<br>

## Figure
![Setting](https://github.com/FANXIN-Gmail/mini-Train/assets/70795543/dfb9bf10-c193-4ae4-bcdc-1516972a7879)
<br><br><br>

## Requirements
torch==1.4.0 <br>
pandas==0.24.2 <br>
scipy==1.3.0 <br>
numpy==1.22.0 <br>
tensorboardX==1.8 <br>
scikit-learn==0.23.2 <br>
tqdm==4.48.2 <br>

# How to start



Run the "train_sample.py" to train the model. <br>
Run the "test_sample.py" to evaluate the model. <br>

