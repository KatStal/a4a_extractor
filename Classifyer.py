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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm, cross_validation, metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
#from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
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
import random
from classifiers_acl_2016 import svm_2
#from helping_func import *
#import gensim
from nlp_preproc import split_into_sentences, tokenizer_split
sns.set()

reload(sys)
sys.setdefaultencoding('utf-8')

category_mapping = {'jo':0, 'pm':1, 'sm':2, 'sc':3}
stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "nevertheless", "next", "nine", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
# "neither", "never", "no", "nobody", "none", "noone", "nor", "not", "nothing"

def data_generator(file1, file2):
	'''Reads in two files: one in the format ID;Sentence;Label. Another: ID;Feature;value'''
	DATA = OrderedDict()
	with codecs.open(file1, 'r', 'utf-8') as raw:
		with codecs.open(file2, 'r', 'utf-8') as features:
			reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
			reader_features = csv.DictReader(features, delimiter=';')
			for row in reader_raw:
				DATA[int(row['ID'])-1]= {'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category'], 'features':{}}
				if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
					print row['ID']
			for row in reader_features:
				if row['value'] == '0':	DATA[int(row['ID'])-1]['features'][row['Feature']]=False
				elif row['value'] == '1': DATA[int(row['ID'])-1]['features'][row['Feature']]=True
	#print DATA[3416]
	#print DATA[8319]
	return DATA

def data_generator_directives(file1, file2):
	'''Reads in two files: one in the format ID;Sentence;Label. Another: ID;Feature;value'''
	DATA = {}
	with codecs.open(file1, 'r', 'utf-8') as raw:
		with codecs.open(file2, 'r', 'utf-8') as features:
			reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
			reader_features = csv.DictReader(features, delimiter=';')
			for row in reader_raw:
				if row['Label'] != "8":
					DATA[int(row['ID'])-1]= {'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category'], 'features':{}}
					if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
						print row['ID']
			for row in reader_features:
				try:
					if row['value'] == '0':	DATA[int(row['ID'])-1]['features'][row['Feature']]=False
					elif row['value'] == '1': DATA[int(row['ID'])-1]['features'][row['Feature']]=True
				except:
					print row['ID'] + ' is not a4a'
	return DATA

def get_labels(data):
	'''Returns a list of labels retrieved from data dictionary produced by data_generator.''' 
	labels = []
	for i in data.keys():
		labels.append(data[i]['label'])
	return np.array(labels)

def correct_labels(labels):
	''' This function takes an np array of all labels as an input,
	and returns an np array with only four labels:
	1 and 3 -> 1
	2,4,5 -> 2
	11 ->
	6,7,9 -> 7'''
	new_labels = []
	for label in labels:
		if label == "3":
			label = "1"
		elif label == "4" or label == "5":
			label = "2"
		elif label == "6" or label == "9":
			label = "7"
		new_labels.append(label)
	return np.array(new_labels)
	
class GetItem(BaseEstimator, TransformerMixin):
	'''
	Returns either a list of raw sentences, or a list of dictionaries with linguistic features and their values, 
	or a list of mapped categories, depending on the key provided: "sent", "features" or "category"
	'''
	def __init__(self, key):
		self.key = key
	
	def fit(self, x, y=None, **fit_params):
		return self

	def transform(self, data):
		items = []
		for i in range(len(data)):
			if self.key == 'sent':
				items.append(data[i][self.key])
			elif self.key == 'features':
				items.append(OrderedDict(sorted(data[i][self.key].items())))
			elif self.key == 'category':
				items.append([category_mapping[data[i][self.key]]])
			elif self.key == 'vector':
				items.append(tokenizer_split(data[i]['sent']))
		return items

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
		
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
		
		
text_clf = Pipeline([('union', FeatureUnion([
							('tfidf', Pipeline([
								('sents', GetItem(key='sent')), 
								('vect', TfidfVectorizer(ngram_range=(1,4))) # stop_words=stopwords
								#('trunc', TruncatedSVD(n_components=10)) # doesn't work with MNB and GNB
								])),
							('LF', Pipeline([
								('features', GetItem(key='features')), 
								('dict_vect', DictVectorizer())
								])),
							('cat', Pipeline([
								('categories', GetItem(key='category')), 
								('ohe', OneHotEncoder())
								]))
							],
							transformer_weights={'LF': 0.6, 'tfidf': 0.8,  'cat':0.4} #, 'cat':0.5
							)),
					#('clf', SGDClassifier(alpha=0.01, loss='log', n_iter=10, penalty='none'))
					('clf', SGDClassifier())
					#('clf', svm.SVC(kernel='linear', C=1))
					#('clf', MultinomialNB())
					])
						
text_clf_2 = Pipeline([('features', GetItem(key='features')), 
						('dict_vect', DictVectorizer()),
						#('clf', MultinomialNB())
						('clf', SGDClassifier())
						#('clf', svm.SVC(kernel='linear', C=1))
						])

text_clf_3 = Pipeline([('sents', GetItem(key='sent')), 
						('tfidf', TfidfVectorizer(ngram_range=(1,4))), #, stop_words='english'
						#('clf', MultinomialNB())
						('clf', SGDClassifier())
						#('clf', svm.SVC(kernel='linear', C=1))
						#('clf', KNeighborsClassifier())
						])						

text_clf_4 = Pipeline([('union', FeatureUnion([
							('tfidf', Pipeline([
								('sents', GetItem(key='sent')), 
								('vect', TfidfVectorizer(ngram_range=(1,1)))
								])),
							('LF', Pipeline([
								('features', GetItem(key='features')), 
								('dict_vect', DictVectorizer())
								]))
							],
							transformer_weights={'LF': 0.6, 'tfidf': 0.8} #, 'cat':0.5
							)),
					#('clf', MultinomialNB())		
					('clf', SGDClassifier())
					])	
text_clf_5 = Pipeline([('union', FeatureUnion([
							('cat', Pipeline([
								('sents', GetItem(key='category')), 
								('ohe', OneHotEncoder())
								])),
							('LF', Pipeline([
								('features', GetItem(key='features')), 
								('dict_vect', DictVectorizer())
								]))
							],
							transformer_weights={'LF': 0.6, 'tfidf': 0.8} #, 'cat':0.5
							)),
					#('clf', MultinomialNB())		
					('clf', SGDClassifier())
					])
	
#f1 = r'E:\Thesis\Development\corpora\Corpus_5000_tab-sep.csv'   #r'E:\Thesis\Development\corpora\TestLFCorpus.txt'	#r'E:\Thesis\Development\corpora\dir_vs_ndir.csv'                      
#f2 = r'E:\Soft\Python\Summer School\MyScripts\corpus_5000_LF.csv'   #r'E:\Thesis\Development\corpora\data_with_features_test.csv' 	#r'E:\Soft\Python\Summer School\MyScripts\dir_vs_ndir_big_train_corpus.csv'  
f1 = r'E:\LMU\Thesis\Development\corpora\Corpus_15K_2Lab.csv'
#f1 = r'E:\Thesis\Development\corpora\corpus_15K_allLab.txt'
f2 = r'E:\LMU\Thesis\Development\corpora\corpus_15K_allLab_LF.csv'
 
if __name__ == '__main__':
	
	#########################################
	#######       GENERATE DATA       #######
	#########################################
	
	all_data = data_generator(f1, f2)
	
	#all_data = data_generator3(r'E:\Thesis\Development\corpora\corpus_5k_allLab.txt')
	labels = get_labels(all_data)
	#print len(labels)
	#print len(labels) == len(all_data)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(all_data, labels, test_size=0.25, random_state=0)
	print "Data is ready"
	#print X_test[0]
	#print len(X_train), len(y_train)
	#print len(X_test), len(y_test)
	
	svm_2.fit(X_train, y_train)
	predicted = svm_2.predict(X_test)
	#score = accuracy_score(y_test, predicted, normalize=False)
	#print 'accuracy 1: ' + str(score)
	#text_clf_4.fit(X_train, y_train)
	#predicted4 = text_clf_4.predict(X_test)
	#score4 = accuracy_score(y_test, predicted4, normalize=False)
	#print 'accuracy 4: ' + str(score4)
	#text_clf_3.fit(X_train, y_train)
	#predicted3 = text_clf_3.predict(X_test)
	#score3 = accuracy_score(y_test, predicted3, normalize=False)
	
	misclassified = (y_test != predicted)
	#test = ["(CCC 2313) Defending one's country against aggression is permitted, but we should never forget that every human life, from the moment of conception, is sacred because it is made in God's image and likeness."]
	#count_vect = CountVectorizer()
	#tfidf_transformer = TfidfTransformer()
	#X_new_counts = count_vect.fit_transform(test)
	#X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)
	#pred_test = text_clf_3.predict(test)
	#print np.where(misclassified)
	#print pred_test
	with codecs.open(r'E:\LMU\Thesis\misclassified.txt', 'w', 'utf8') as miscl:
		for a in np.where(misclassified):
			for e in a:
				miscl.write(str(X_test[e]))
				miscl.write('\n')
	
	#print 'accuracy 3: ' + str(score3)
	#text_clf_5.fit(X_train, y_train)
	#predicted5 = text_clf_5.predict(X_test)
	#score5 = accuracy_score(y_test, predicted5, normalize=False)
	#print 'accuracy 5: ' + str(score5)
	
	#svm_tfidf_cat.fit(X_train, y_train)
	#predictedXX = svm_tfidf_cat.predict(X_test)
	#scoreXX = accuracy_score(y_test, predictedXX, normalize=False)
	#print 'accuracy XX: ' + str(scoreXX)
	
	parameters = {'union__tfidf__vect__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4)),
				#'union__tfidf__trunc__n_components': (range(100, 4500, 50)),	
				'clf__alpha': list(10.0**-np.arange(1,7)), # Param for SGD
				'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'), # Param for SGD
				'clf__penalty': ('none', 'l2', 'l1', 'elasticnet'), # Param for SGD
				'clf__n_iter': (5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15), # Param for SGD
				'clf__random_state':(None, 10, 20, 30, 40, 50)} # Param for SGD
                #'clf__alpha': list(10.0**-np.arange(1,7)), # Param for MNB
                #'clf__fit_prior': (True, False), # Param for MNB
                #'clf__class_prior': ([.5, .5], [.7, .3], [.3, .7], [.6, .4], [.4, .6], [.8, .2], [.2, .8])} # Param for MNB
				
	#text_clf_3.fit(all_data, labels)
	#joblib.dump(clf, 'DirVSNdir_SGD_allfeat.pkl', compress=9)
	
	#xx = GetItem(key='vector')
	#xs = xx.transform(all_data)
	#print str(xs)
	#model = gensim.models.Word2Vec(xs, size=100)
	#print 'model done'
	#w2v = dict(zip(model.index2word, model.syn0))
						
	#etree_w2v = Pipeline([
	#("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
	#("extra trees", ExtraTreesClassifier(n_estimators=200))])
	
	#####################################################
	#######      GET CROSS-VALIDATION SCORES      #######
	#####################################################
	
	#scores = cross_validation.cross_val_score(text_clf,  all_data, labels, cv=5)
	#scores2 = cross_validation.cross_val_score(etree_w2v, all_data, labels, cv=5)
	#scores3 = cross_validation.cross_val_score(text_clf_3, all_data, labels, cv=5)

	#print 'clf', scores
	#print 'clf 2', scores2
	#print 'clf 3', scores3
	
	
	############################################################
	####### CLASSIFICATION SCORES: PRECISION, RECALL, F1 #######
	############################################################
		
	#text_clf.fit(X_train, y_train)
	#etree_w2v.fit(X_train, y_train)
	# text_clf_2.fit(X_train, y_train)
	#text_clf_3.fit(X_train, y_train)
	# text_clf_4.fit(X_train, y_train)
	#text_clf_5.fit(X_train, y_train)
	#svm_tfidf_cat.fit(X_train, y_train)
	
	#predicted = text_clf.predict(X_test)
	#predicted2 = etree_w2v.predict(X_test)
	#predicted3 = text_clf_3.predict(X_test)
	# predicted4 = text_clf_4.predict(X_test)
	#predicted5 = text_clf_5.predict(X_test)
	#predictedXX = svm_tfidf_cat.predict(X_test)
	
	#print '3 groups of features', (metrics.classification_report(y_test, predicted))
	#print 'w2v', (metrics.classification_report(y_test, predicted2))
	# print '2 groups of features', (metrics.classification_report(y_test, predicted4))
	#print 'tf-idf', (metrics.classification_report(y_test, predicted3))
	# print 'LF', (metrics.classification_report(y_test, predicted2))
	# print 'LF+cat', (metrics.classification_report(y_test, predicted5))
	#print 'tf-idf+cat', (metrics.classification_report(y_test, predictedXX))
	
	#################################################################################
	#########  Uncomment below to perform grid search for best parameters  ##########
	#################################################################################
	
	# gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
	
	# print("Performing grid search...")
	# print("pipeline:", [name for name, _ in text_clf.steps])
	# print("parameters:")
	# pprint(parameters)
	
	# gs_clf = gs_clf.fit(all_data, labels)
	
	# print("Best score: %0.3f" % gs_clf.best_score_)
	
	# best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
	# for param_name in sorted(parameters.keys()):
		# print("%s: %r" % (param_name, best_parameters[param_name]))
		
		
	##############################################
	########## LEARNING CURVES ###################
	##############################################
	
	#print "Getting curves..."
	'''Linguistic features + tf-idf'''
	two_groups = FeatureUnion([('tfidf', Pipeline([('sents', GetItem(key='sent')),('vect', TfidfVectorizer())])),('LF', Pipeline([('features', GetItem(key='features')),('dict_vect', DictVectorizer())]))])
	all_feat = two_groups.fit_transform(all_data)
	#print "LF+TFIDF"
	'''Linguistic features + tf-idf + category'''
	three_groups = FeatureUnion([('tfidf', Pipeline([('sents', GetItem(key='sent')),('vect', TfidfVectorizer(ngram_range=(1,4)))])),('LF', Pipeline([('features', GetItem(key='features')),('dict_vect', DictVectorizer())])),('cat', Pipeline([('cats', GetItem(key='category')),('ohe', OneHotEncoder())]))])
	all_feat_2 = three_groups.fit_transform(all_data)
	#print "LF+TFIDF+Category"
	'''tf-idf'''
	xx = GetItem(key='sent')
	xs = xx.transform(all_data)
	tf = TfidfVectorizer(ngram_range=(1,4))
	fr = tf.fit_transform(xs)
	#print "TF-IDF"
	# '''Linguistic features'''
	# xxx = GetItem(key='features')
	# xss = xxx.transform(all_data)
	# bin = DictVectorizer()
	# lf = bin.fit_transform(xss)
	
	# #names = bin.get_feature_names()
	
	est = SGDClassifier()
	
	title = "Learning Curves (LF & TF-IDF)"
	title2 = "Learning Curves (TF-IDF)"
	# title3 = "Learning Curves (LF)"
	title4 = "Learning Curves (LF & TF-IDF & Data origin)"
	#sp20 = " 20% split"
	sp25 = " 25% split"
	# sp30 = " 30% split"
	
	# # Cross validation with 100 iterations to get smoother mean test and train
	# # score curves, each time with 20% data randomly selected as a validation set.
	
	#cv = cross_validation.ShuffleSplit(5000, n_iter=100, test_size=0.2, random_state=0)
	cv2 = cross_validation.ShuffleSplit(5000, n_iter=100, test_size=0.25, random_state=0)
	# cv3 = cross_validation.ShuffleSplit(5000, n_iter=100, test_size=0.3, random_state=0)
	
	#plot_learning_curve(est, title+sp20, all_feat, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	#plot_learning_curve(est, title2+sp20, fr, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	# plot_learning_curve(est, title3+sp20, lf, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	#plot_learning_curve(est, title4+sp20, all_feat_2, labels, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
	
	#plot_learning_curve(est, title+sp25, all_feat, labels, ylim=(0.7, 1.01), cv=cv2, n_jobs=4)
	#plot_learning_curve(est, title2+sp25, fr, labels, ylim=(0.7, 1.01), cv=cv2, n_jobs=4)
	# plot_learning_curve(est, title3+sp25, lf, labels, ylim=(0.7, 1.01), cv=cv2, n_jobs=4)
	#plot_learning_curve(est, title4+sp25, all_feat_2, labels, ylim=(0.7, 1.01), cv=cv2, n_jobs=4)
	
	# plot_learning_curve(est, title+sp30, all_feat, labels, ylim=(0.7, 1.01), cv=cv3, n_jobs=4)
	# plot_learning_curve(est, title2+sp30, fr, labels, ylim=(0.7, 1.01), cv=cv3, n_jobs=4)
	# plot_learning_curve(est, title3+sp30, lf, labels, ylim=(0.7, 1.01), cv=cv3, n_jobs=4)
	# plot_learning_curve(est, title4+sp30, all_feat_2, labels, ylim=(0.7, 1.01), cv=cv3, n_jobs=4)
	#plt.show()
	
	
	################################################################
	#################### DATA VISUALIZATION ########################
	################################################################

	#df = DataFrame(np.hstack((frd[:20,:20], nl[:20, None])),columns = (list(range(frd[:20,:20].shape[1])) + ["class"]))
	#sns.pairplot(df, hue="class", size=1.5)
	#.savefig(r'E:\Soft\Python\Summer School\MyScripts\fig1.png')
	#plt.spy(fr)
	#plt.plot(fr, labels)
	
	
	