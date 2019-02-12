#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Run it with ipython
'''

import codecs
import csv
import numpy as np
from random import shuffle
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import svm, cross_validation, metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction import DictVectorizer
#from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import Binarizer
from pprint import pprint
from sklearn.grid_search import GridSearchCV
import sys
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
import seaborn as sns
#from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
#import scipy.sparse as sps
#%matplotlib inline
from classifiers_acl_2016 import svm_2, svm_tfidf_cat
from helping_func import *
sns.set()

reload(sys)
sys.setdefaultencoding('utf-8')

# def data_generator(file1, file2):
	# '''Reads in two files: one in the format ID;Sentence;Label. Another: ID;Feature;value'''
	# DATA = {}
	# with codecs.open(file1, 'r', 'utf-8') as raw:
		# with codecs.open(file2, 'r', 'utf-8') as features:
			# reader_raw = csv.DictReader(raw, delimiter=',', quotechar='"') #';' for TestLFCorpus, ',' for dir_vs_ndir, '\t' for big_training_corpus_5k
			# reader_features = csv.DictReader(features, delimiter=';', quotechar='|')
			# for row in reader_raw:
				# DATA[int(row['ID'])]= {'sent':row['Sentence'], 'label':row['Label'],  'features':{}}
			# for row in reader_features:
				# if row['value'] == '0':	DATA[int(row['ID'])]['features'][row['Feature']]=False
				# elif row['value'] == '1': DATA[int(row['ID'])]['features'][row['Feature']]=True
	# return DATA			

def data_generator_2Lab(file1, file2):
	'''Reads in two files: one in the format ID;Sentence;Label. Another: ID;Feature;value'''
	import codecs
	import csv
	DATA = {}
	with codecs.open(file1, 'r', 'utf-8') as raw:
		with codecs.open(file2, 'r', 'utf-8') as features:
			reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
			reader_features = csv.DictReader(features, delimiter=';')
			for row in reader_raw:
				DATA[int(row['ID'])-1]= {'sent':row['Sentence'], 'label':row['Label'], 'category':'jo', 'features':{}}
				if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
					print row['ID']
			for row in reader_features:
				if row['value'] == '0':	DATA[int(row['ID'])-1]['features'][row['Feature']]=False
				elif row['value'] == '1': DATA[int(row['ID'])-1]['features'][row['Feature']]=True
	return DATA	
	
def data_generator_predict(file1, file2):
	'''Reads in two files: one in the format ID;Sentence;Label. Another: ID;Feature;value'''
	DATA = {}
	with codecs.open(file1, 'r', 'utf-8') as raw:
		with codecs.open(file2, 'r', 'utf-8') as features:
			reader_features = csv.DictReader(features, delimiter=';', quotechar='|')
			for row in raw.readlines()[1:]:
				id=row[:row.find(',')]
				sent= row[row.find(',')+1:row.find(',???,')]
				#sent=sent[:sent.find(',???')]
				DATA[int(id)-1]= {'sent':sent, 'features':{}}
			for row in reader_features:
				try:
					if row['value'] == '0':	DATA[int(row['ID'])-1]['features'][row['Feature']]=False
					elif row['value'] == '1': DATA[int(row['ID'])-1]['features'][row['Feature']]=True
				except:
					print 'Not a4a!'
	return DATA

def data_generator_predict_noLF(file1):
	'''Reads in two files: one in the format ID;Sentence;Label. Another: ID;Feature;value'''
	DATA = {}
	with codecs.open(file1, 'r', 'utf-8') as raw:
		#with codecs.open(file2, 'r', 'utf-8') as features:
		#	reader_features = csv.DictReader(features, delimiter=';', quotechar='|')
		for row in raw.readlines()[1:]:
			id=row[:row.find(',')]
			sent= row[row.find(',')+1:row.find(',???,')]
				#sent=sent[:sent.find(',???')]
			DATA[int(id)]= {'sent':sent, 'category':'jo'}
			#for row in reader_features:
			#	try:
			#		if row['value'] == '0':	DATA[int(row['ID'])]['features'][row['Feature']]=False
			#		elif row['value'] == '1': DATA[int(row['ID'])]['features'][row['Feature']]=True
			#	except:
			#		print 'Not a4a!'
	return DATA			

def get_labels(data):
	'''Returns a list of labels retrieved from data dictionary produced by data_generator.''' 
	labels = []
	for i in data:
		labels.append(data[i]['label'])
	return np.array(labels)

#print type(get_labels(DATA))
#print get_labels(DATA)

class GetItem(BaseEstimator, TransformerMixin):
	'''
	Returns either a list of raw sentences, or a list of dictionaries with linguistic 
	features and their values depending on the key provided: "sent" or "features"
	'''
	def __init__(self, key):
		self.key = key
	
	def fit(self, x, y=None, **fit_params):
		return self

	def transform(self, data):
		items = []
		
		for i in data:
			if self.key == 'sent':
				items.append(data[i][self.key])
			elif self.key == 'features':
				items.append(OrderedDict(sorted(data[i][self.key].items())))
			elif self.key == 'category':
				try:
					items.append([category_mapping[data[i][self.key]]])
				except:
					print data[i]
	
		return items

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
		"""
		Generate a simple plot of the test and traning learning curve.

		Parameters
		----------
		estimator : object type that implements the "fit" and "predict" methods
			An object of that type which is cloned for each validation.

		title : string
			Title for the chart.

		X : array-like, shape (n_samples, n_features)
			Training vector, where n_samples is the number of samples and
			n_features is the number of features.

		y : array-like, shape (n_samples) or (n_samples, n_features), optional
			Target relative to X for classification or regression;
			None for unsupervised learning.

		ylim : tuple, shape (ymin, ymax), optional
			Defines minimum and maximum yvalues plotted.

		cv : integer, cross-validation generator, optional
			If an integer is passed, it is the number of folds (defaults to 3).
			Specific cross-validation objects can be passed, see
			sklearn.cross_validation module for the list of possible objects

		n_jobs : integer, optional
			Number of jobs to run in parallel (default 1).
		"""
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()

		plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
						 train_scores_mean + train_scores_std, alpha=0.1,
						 color="r")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
						 test_scores_mean + test_scores_std, alpha=0.1, color="g")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

		plt.legend(loc="best")
		return plt

f1 = r'E:\Thesis\Development\corpora\Corpus_15K_2Lab.csv'                         
f2 = r'E:\Thesis\Development\corpora\corpus_15K_LF.csv'
f3 = r'E:\Thesis\Development\corpora\Corpus_15K_allLab.txt'

nyt_predict = r"C:\Users\stolpovskaya\Dropbox\thesis\NYT\NYT_predict_new.txt"
nyt_lf = r"C:\Users\stolpovskaya\Dropbox\thesis\NYT\NYT_LF_new.txt"
guardian_predict = r"C:\Users\stolpovskaya\Dropbox\thesis\Guardian\Guardian_predict_new.txt"
guardian_lf = r"C:\Users\stolpovskaya\Dropbox\thesis\Guardian\Guardian_LF_new.txt"

if __name__ == '__main__':
	all_data_2Lab = data_generator_2Lab(f1, f2)
	all_data_7Lab = data_generator_7Lab(f3)
	print 'Data ready!'
	
	labels = get_labels(all_data_2Lab)
	labels_all = get_labels(all_data_7Lab)
	print 'Training...'
	
	svm_2.fit(all_data_2Lab, labels)
	svm_tfidf_cat.fit(all_data_7Lab, labels_all)
	print 'Done training'
	#joblib.dump(clf, 'DirVSNdir_SGD_allfeat.pkl', compress=9)
	#scores = cross_validation.cross_val_score(text_clf,  all_data, labels, cv=5)
	#scores2 = cross_validation.cross_val_score(text_clf_2, all_data, labels, cv=5)
	#scores3 = cross_validation.cross_val_score(text_clf_3, all_data, labels, cv=5)

	#print 'clf', scores
	#print 'clf 2', scores2
	#print 'clf 3', scores3

##############################################
########## LEARNING CURVES ###################
##############################################
	
	# x = FeatureUnion([('tfidf', Pipeline([('sents', GetItem(key='sent')),('vect', TfidfVectorizer(ngram_range=(1, 3)))])),('LF', Pipeline([('features', GetItem(key='features')),('dict_vect', DictVectorizer())]))])
	# all_feat = x.fit_transform(all_data)
	
	# xx = GetItem(key='sent')
	# xs = xx.transform(all_data)
	# tf = TfidfVectorizer(ngram_range=(1, 4))
	# fr = tf.fit_transform(xs)
	# xxx = GetItem(key='features')
	# xss = xxx.transform(all_data)
	# bin = DictVectorizer()
	# lf = bin.fit_transform(xss)
	# #names = bin.get_feature_names()
	
	# est = MultinomialNB(fit_prior=True, alpha=0.1000000000000001)
	# title = "Learning Curves for Combined Features"
	# title2 = "Learning Curves for tf-idf"
	# title3 = "Learning Curves for Linguistic Features"
	
	# # Cross validation with 100 iterations to get smoother mean test and train
	# # score curves, each time with 20% data randomly selected as a validation set.
	# cv = cross_validation.ShuffleSplit(all_feat.shape[0], n_iter=100, test_size=0.2, random_state=0)
	# cv2 = cross_validation.ShuffleSplit(fr.shape[0], n_iter=100, test_size=0.2, random_state=0)
	# cv3 = cross_validation.ShuffleSplit(lf.shape[0], n_iter=100, test_size=0.2, random_state=0)

	# plot_learning_curve(est, title, all_feat, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	# plot_learning_curve(est, title2, fr, labels, ylim=(0.7, 1.01), cv=cv2, n_jobs=4)
	# plot_learning_curve(est, title3, lf, labels, ylim=(0.7, 1.01), cv=cv3, n_jobs=4)
	# plt.show()
	
############################################################
#################### SYRIAN CW #############################
############################################################
	with codecs.open(r'C:\Users\stolpovskaya\Dropbox\thesis\NYT\NYT_res_2Lab_1.txt', 'w', 'utf8') as result_2lab:
		predict_data_2Lab = data_generator_predict(nyt_predict, nyt_lf)
		print 'data for NYT generated'
		#for a, b in predict_data_2Lab.items():
		#	result_2lab.write(str(a) + '\t' + str(b) + '\n')
		predicted = svm_2.predict(predict_data_2Lab)
		#with codecs.open(r'C:\Users\stolpovskaya\Dropbox\thesis\NYT\NYT_a4a.txt', 'w', 'utf8') as a4a:
		for id, lab in zip(predict_data_2Lab, predicted):
			result_2lab.write(str(id)+'\t'+lab+'\n')
				#if lab == 'dir':
					
		
	### fine-grained classification ###
	nyt_positives = get_a4a(r'C:\Users\stolpovskaya\Dropbox\thesis\NYT\NYT_res_2Lab.txt', nyt_predict)
	predict_data_7Lab = correct_id(data_generator_predict_noLF(nyt_positives), 'NYT')
	predicted_7Lab  = svm_tfidf_cat.predict(predict_data_7Lab)
	with codecs.open(r'C:\Users\stolpovskaya\Dropbox\thesis\NYT\NYT_res_7Lab.txt', 'w', 'utf8') as res_7Lab:
		for id, lab in zip(predict_data_7Lab, predicted_7Lab):
			res_7Lab.write(str(id+1)+'\t'+lab+'\n')
	print 'Done for NYT'			

	with codecs.open(r'C:\Users\stolpovskaya\Dropbox\thesis\Guardian\Guardian_res_2Lab.txt', 'w', 'utf8') as result2:
		predict_data2 = data_generator_predict(guardian_predict, guardian_lf)
		print 'data for Guardian generated'
		predicted2 = svm_2.predict(predict_data2)
		for id, lab in zip(predict_data2, predicted2):
			result2.write(str(id+1)+'\t'+lab+'\n')

	### fine-grained classification ###
	guardian_positives = get_a4a(r'C:\Users\stolpovskaya\Dropbox\thesis\Guardian\Guardian_res_2Lab.txt', nyt_predict)
	predict_data_7Lab_2 = correct_id(data_generator_predict_noLF(guardian_positives), 'Guardian')
	predicted_7Lab_2  = svm_tfidf_cat.predict(predict_data_7Lab_2)
	with codecs.open(r'C:\Users\stolpovskaya\Dropbox\thesis\Guardian\Guardian_res_7Lab.txt', 'w', 'utf8') as res_7Lab2:
		for id, lab in zip(predict_data_7Lab_2, predicted_7Lab_2):
			res_7Lab2.write(str(id+1)+'\t'+lab+'\n')			
	print 'Done for Guardian'