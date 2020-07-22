# -*- coding: utf-8 -*-

from gensim.models import word2vec
from gensim import models
import logging
import pickle 
import numpy as np

tfidf=pickle.load(open("min_tfidf.pickle", "rb" ))
X=pickle.load(open("min_X.pickle", "rb" ))
# print(X.shape)
# print(min_X.shape)

vocab=tfidf.vocabulary_
feature_names=tfidf.get_feature_names()

def kcm(word,e):
	if word in vocab:
		i=vocab[word]
		doc_arr=X.T[i].toarray()[0]
		doc_id_arr=(-doc_arr).argsort()
		
		top_k=800
		n=0
		bag=[word]

		for doc_id in doc_id_arr:
			if n>=top_k:
				break
			# print(doc_arr[doc_id])
			if doc_arr[doc_id] < 1 and doc_arr[doc_id] >0:
				data=X.data[X.indptr[doc_id]:X.indptr[doc_id+1]]
				index=X.indices[X.indptr[doc_id]:X.indptr[doc_id+1]]
				zip_list=[z for z in zip(data,index) if z[0]<1 and z[0]>0]
				w_id=sorted(zip_list,key = lambda z: z[0],reverse=True)
				# w_id = [x[1] for x in list1][:40]
				for _id in w_id :
					# if arrs[_id] == 0:
					# 	break
					if feature_names[_id[1]] not in e and _id[0]>0.05:
						bag.append(feature_names[_id[1]])
						# print(feature_names[_id],arrs[_id])
						n+=1
						# print(n)
						if n>=top_k:
							break
		return bag
	return []

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	# model = models.Word2Vec.load('word2vec.model')
	model = models.Word2Vec.load('word2vec_comma.model')

	while True:
		try:
			query = input('請輸入:')
			S = {}
			# print("相似詞前 100 排序")
			topn=15
			res = model.most_similar(query,topn = topn)
			e=[e[0] for e in res]
			e.append(query)
			print('同義詞')
			for ew in res:
				bag=kcm(ew[0],e)
				print(topn,ew[0])
				topn-=1
				for cw in bag:
					if cw in S:
						S[cw]=S[cw]+1
					else:
						S[cw]=1
					# print(cw,S[cw])
			print()
			print(query,"是:")
			edm = sorted(S.items(),key=lambda item: item[1],reverse=True)

			for i in range(20):
				print(edm[i])

		except Exception as e:
			print(repr(e))

if __name__ == "__main__":
	main()
