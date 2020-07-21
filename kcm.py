from sklearn.feature_extraction.text import TfidfVectorizer
import pickle




with open('wiki_seg_to_doc.txt', 'r', encoding='utf-8') as content :
    docs=content.read().split('\n')
    print(len(docs))



tfidf = TfidfVectorizer(min_df=2)
X=tfidf.fit_transform(docs)
# print(tfidf.vocabulary_)

pickle.dump(tfidf, open("min_tfidf.pickle", "wb"))
pickle.dump(X, open("min_X.pickle", "wb"))


# pickle.dump(tfidf.fit_transform(docs).toarray().T, open("arr.pickle", "wb"))
# pickle.dump(tfidf.get_feature_names(), open("feature_name_test.pickle", "wb"))
# pickle.dump(tfidf.vocabulary_, open("vocab_test.pickle", "wb"))




