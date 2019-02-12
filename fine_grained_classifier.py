#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Run it with ipython
'''

import codecs
import csv
import numpy as np
from random import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import cross_validation, metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, make_scorer, precision_recall_fscore_support
from sklearn.preprocessing import Binarizer, OneHotEncoder
from pprint import pprint
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier
import sys
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.model_selection import cross_val_score
import seaborn as sns
from helping_func import *
sns.set()

reload(sys)
sys.setdefaultencoding('utf-8')

stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "nevertheless", "next", "nine", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

#################### Naive Bayes ###########################

nb_tfidf = Pipeline([('sents', GetItem(key='sent')), 
						('tfidf', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords)),
						('clf', MultinomialNB())
					])						
					
nb_tfidf_cat = Pipeline([('union', FeatureUnion([
							('tfidf', Pipeline([
								('sents', GetItem(key='sent')), 
								('vect', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords))
								])),
							('cat', Pipeline([
								('categories', GetItem(key='category')), 
								('ohe', OneHotEncoder())
								]))
							],
							transformer_weights={'tfidf': 0.8,  'cat':0.4} #, 'cat':0.5
							)),
					('clf', MultinomialNB())
					])

###################### SVM ####################################			

svm_tfidf = Pipeline([('sents', GetItem(key='sent')), 
						('tfidf', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords)),
						('clf', SGDClassifier())
						])						

svm_tfidf_cat = Pipeline([('union', FeatureUnion([
							('tfidf', Pipeline([
								('sents', GetItem(key='sent')), 
								('vect', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords))
								])),
							
							('cat', Pipeline([
								('categories', GetItem(key='category')), 
								('ohe', OneHotEncoder())
								]))
							],
							transformer_weights={'tfidf': 0.8,  'cat':0.4} #, 'cat':0.5
							)),
					('clf', SGDClassifier())
					])					
######################## KNN #####################################

knn_tfidf = Pipeline([('sents', GetItem(key='sent')), 
						('tfidf', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords)),
						('clf', KNeighborsClassifier())
						])						

knn_tfidf_cat = Pipeline([('union', FeatureUnion([
							('tfidf', Pipeline([
								('sents', GetItem(key='sent')), 
								('vect', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords))
								])),
							('cat', Pipeline([
								('categories', GetItem(key='category')), 
								('ohe', OneHotEncoder())
								]))
							],
							transformer_weights={'tfidf': 0.8,  'cat':0.4} #, 'cat':0.5
							)),
					('clf', KNeighborsClassifier())
					])

######################## Decision Tree ##################################

dt_tfidf = Pipeline([('sents', GetItem(key='sent')), 
						('tfidf', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords)),
						('clf', tree.DecisionTreeClassifier())
						])						

dt_tfidf_cat = Pipeline([('union', FeatureUnion([
							('tfidf', Pipeline([
								('sents', GetItem(key='sent')), 
								('vect', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords))
								])),
							('cat', Pipeline([
								('categories', GetItem(key='category')), 
								('ohe', OneHotEncoder())
								]))
							],
							transformer_weights={'tfidf': 0.8,  'cat':0.4} #, 'cat':0.5
							)),
					('clf', tree.DecisionTreeClassifier())
					])
					
############################# MLPClassifier #################################################

mlp_tfidf = Pipeline([('sents', GetItem(key='sent')), 
						('tfidf', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords)),
						('clf', MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
						])						

mlp_tfidf_cat = Pipeline([('union', FeatureUnion([
							('tfidf', Pipeline([
								('sents', GetItem(key='sent')), 
								('vect', TfidfVectorizer(ngram_range=(1,4), stop_words=stopwords))
								])),
							('cat', Pipeline([
								('categories', GetItem(key='category')), 
								('ohe', OneHotEncoder())
								]))
							],
							transformer_weights={'tfidf': 0.8,  'cat':0.4} #, 'cat':0.5
							)),
					('clf', MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
					])
f = r'E:\LMU\Thesis\Development\corpora\Corpus_15K_allLab.txt'

if __name__ == '__main__':

	#########################################
	#######       GENERATE DATA       #######
	#########################################
	
	#all_data = notdoing_vs_multiple_vs_general_vs_other(f)
	all_data = notdoing_vs_other(f)
	labels = get_labels(all_data)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(all_data, labels, test_size=0.20, random_state=0)
	
	####################################################################
	########## CLASSIFICATION RESULTS FOR 15K CORPUS ALL LABELS ##########
	####################################################################
		
	with open(r"E:\LMU\Thesis\Development\Classification_results\clf_results_15K_notDoing_vs_other_20split.txt", 'w') as res:
		res.write('Fine grained classification as agendas for not doing (11) vs. other (6+7+9), other labels (1-5) were excluded.\n\n')
		
		'''Naive Bayes'''
		print "Naive Bayes"
		
		nb_tfidf.fit(X_train, y_train)
		predicted3 = nb_tfidf.predict(X_test)
		res.write('TF-IDF for NB' + '\n' + str(metrics.classification_report(y_test, predicted3))+ '\n\n\n')
				
		nb_tfidf_cat.fit(X_train, y_train)
		predicted4 = nb_tfidf_cat.predict(X_test)
		res.write('TF-IDF + category for NB' + '\n' + str(metrics.classification_report(y_test, predicted4))+ '\n\n\n')
		
		'''SVM'''
		print "SVM"
		
		svm_tfidf.fit(X_train, y_train)
		predicted7 = svm_tfidf.predict(X_test)
		res.write('TF-IDF for SVM' + '\n' + str(metrics.classification_report(y_test, predicted7))+ '\n\n\n')
		
		svm_tfidf_cat.fit(X_train, y_train)
		predictedXX = svm_tfidf_cat.predict(X_test)
		res.write('2 groups of features (tfidf, category) for SVM' + '\n' + str(metrics.classification_report(y_test, predictedXX))+ '\n\n\n')
		
		''' KNN '''
		print "KNN"
		
		knn_tfidf.fit(X_train, y_train)
		predicted11 = knn_tfidf.predict(X_test)
		res.write('TF-IDF for KNN' + '\n' + str(metrics.classification_report(y_test, predicted11))+ '\n\n\n')
		
		knn_tfidf_cat.fit(X_train, y_train)
		predicted12 = knn_tfidf_cat.predict(X_test)
		res.write('2 groups of features (tfidf, category) for KNN' + '\n' + str(metrics.classification_report(y_test, predicted12))+ '\n\n\n')
		
		''' Decision Tree '''
		print "Decision Tree"
		
		dt_tfidf.fit(X_train, y_train)
		predicted15 = dt_tfidf.predict(X_test)
		res.write('TF-IDF for Decision Tree' + '\n' + str(metrics.classification_report(y_test, predicted15))+ '\n\n\n')
		
		dt_tfidf_cat.fit(X_train, y_train)
		predicted16 = dt_tfidf_cat.predict(X_test)
		res.write('2 groups of features (tfidf, category) for Decision Tree' + '\n' + str(metrics.classification_report(y_test, predicted16))+ '\n\n\n')
		
		''' MLP '''
		print "MLP"
		
		mlp_tfidf.fit(X_train, y_train)
		predicted15 = mlp_tfidf.predict(X_test)
		res.write('TF-IDF for MLP' + '\n' + str(metrics.classification_report(y_test, predicted15))+ '\n\n\n')
				
		mlp_tfidf_cat.fit(X_train, y_train)
		predicted16 = mlp_tfidf_cat.predict(X_test)
		res.write('2 groups of features (tfidf, category) for MLP' + '\n' + str(metrics.classification_report(y_test, predicted16))+ '\n\n\n')					