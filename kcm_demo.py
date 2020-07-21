import pickle 
import numpy as np

# tfidf=pickle.load(open("tfidf_test.pickle", "rb" ))
# X=pickle.load(open("X_test.pickle", "rb" ))
tfidf=pickle.load(open("min_tfidf.pickle", "rb" ))
X=pickle.load(open("min_X.pickle", "rb" ))
X_T=X.T

vocab=tfidf.vocabulary_
feature_names=tfidf.get_feature_names()


def input_w():
    while True:
        word=input('請輸入:')
    # print(X.sahpe)
        if word not in vocab:
            print('dic沒有這個單字')
        else:
            i=vocab[word]
            doc_arr=X.T[i].toarray()[0]
            doc_id_arr=(-doc_arr).argsort()
            
            # data=X_T.data[X_T.indptr[i]:X_T.indptr[i+1]]
            # index=X_T.indices[X_T.indptr[i]:X_T.indptr[i+1]]
            # zip_list=[z for z in zip(data,index) if z[0]<1]
            # list1=sorted(zip_list,key = lambda z: z[0],reverse=True)
            # doc_id_arr = [x[1] for x in list1]
            
            top_k=30
            n=0
            bag=[word]

            for doc_id in doc_id_arr:
                if n>=top_k:
                    break
                # print(doc_arr[doc_id])
                if doc_arr[doc_id] < 1 and doc_arr[doc_id] >0:
                    # arrs=X[doc_id].toarray()[0]
                    # w_id = (-arrs).argsort()[:80]
                    data=X.data[X.indptr[doc_id]:X.indptr[doc_id+1]]
                    index=X.indices[X.indptr[doc_id]:X.indptr[doc_id+1]]
                    zip_list=[z for z in zip(data,index) if z[0]<1]
                    list1=sorted(zip_list,key = lambda z: z[0],reverse=True)
                    w_id = [x[1] for x in list1]
                    for _id in w_id :
                        # if arrs[_id] == 0:
                        #     break
                        if feature_names[_id] not in bag:
                            print(feature_names[_id])
                            n+=1
                            # print(n)
                            if n>=top_k:
                                break
            
            print()

input_w()