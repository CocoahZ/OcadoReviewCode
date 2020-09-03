from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.decomposition import LatentDirichletAllocation


def LDA(content, max_features, topic_num, max_iter):
    # extract features
    tf_vectorizer = CountVectorizer(max_df=400, 
                                    min_df=50, 
                                    max_features=max_features, 
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(content)
    # lda
    lda_model = LatentDirichletAllocation(n_components=topic_num, 
                                          max_iter=max_iter,
                                          learning_method='batch')
    lda_model.fit(tf)
    return lda_model, tf_vectorizer, tf


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))