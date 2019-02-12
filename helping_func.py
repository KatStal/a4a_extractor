from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from collections import OrderedDict
from os.path import basename, dirname
import codecs

def data_generator_2Lab(file1, file2):
	'''Reads in two files: one in the format ID;Sentence;Label. Another: ID;Feature;value
	It splits all data into two categories: not a4a (8), a4a (1,2,3,4,5,6,7,9,11)'''
	import codecs
	import csv
	DATA = {}
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
	return DATA

def data_generator_allLab(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] != '8':
				sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA

def data_generator_4Lab(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Merges labels 1 and 3 into 1; 2,4 and 5 into 2; 9, 7 and 6 into 7. Creates data with labels 'cooperative treatment'(1), restrictive treatment(2), agendas for not doing (11), other(7)
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] != '8':
				if row['Label'] == '3':
					sents.append({'sent':row['Sentence'], 'label':'1', 'category':row['Category']})
				elif row['Label'] == '4' or row['Label'] == '5':
					sents.append({'sent':row['Sentence'], 'label':'2', 'category':row['Category']})
				elif row['Label'] == '9' or row['Label'] == '6':
					sents.append({'sent':row['Sentence'], 'label':'7', 'category':row['Category']})
				
				else:
					sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA

def data_generator_3Lab(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Merges labels 1 and 3 into 1; 2,4 and 5 into 2; 11, 9, 7 and 6 into 7. reates data with labels 'cooperative treatment'(1), restrictive treatment(2), other(7)
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] != '8':
				if row['Label'] == '3':
					sents.append({'sent':row['Sentence'], 'label':'1', 'category':row['Category']})
				elif row['Label'] == '4' or row['Label'] == '5':
					sents.append({'sent':row['Sentence'], 'label':'2', 'category':row['Category']})
				elif row['Label'] == '9' or row['Label'] == '6' or row['Label'] == '11':
					sents.append({'sent':row['Sentence'], 'label':'7', 'category':row['Category']})
				
				else:
					sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA
	
def data_generator_8Lab(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Merges labels 4 and 5 into 4. 
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] != '8':
				if row['Label'] == '5':
					sents.append({'sent':row['Sentence'], 'label':'4', 'category':row['Category']})
				
				else:
					sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA	
	
def data_generator_7Lab(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Merges labels 4 and 5 into 4; 6 and 7 into 7.
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] != '8':
				if row['Label'] == '5':
					sents.append({'sent':row['Sentence'], 'label':'4', 'category':row['Category']})
				elif row['Label'] == '6':
					sents.append({'sent':row['Sentence'], 'label':'7', 'category':row['Category']})
				else:
					sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA		

def deesc_vs_support(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Leaves out all the sentenes except those with labels 1 and 3 (cooperative treatment). Used to classify sents within the category positive treatment. 
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] == '1' or row['Label'] == '3':
				sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA
	
def esc_vs_4_plus_5(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Leaves out all the sentenes except those with labels 2, 4 and 5 (restrictive treatment). 4 and 5 are merged together. Used to classify sents within the category restrictive treatment. 
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] == '2' or row['Label'] == '4' or row['Label'] == '5':
				if row['Label'] == '4' or row['Label'] == '5':
					sents.append({'sent':row['Sentence'], 'label':'4', 'category':row['Category']})
				else:
					sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA		
	
def esc_vs_punish_vs_ignore(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Leaves out all the sentenes except those with labels 2, 4 and 5 (restrictive treatment). Used to classify sents within the category restrictive treatment with 3 different labels.
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] == '2' or row['Label'] == '4' or row['Label'] == '5':
				sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA	

def punish_vs_ignore(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Leaves out all the sentenes except those with labels 4 and 5.
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] == '4' or row['Label'] == '5':
				sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA	

def notdoing_vs_other(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Leaves out all the sentenes except those with labels 11, 6, 7 and 9 (other). Used to classify sents within the category other as a4 not doing (11) vs. other (6+7+9). 
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] == '6' or row['Label'] == '7' or row['Label'] == '9' or row['Label'] == '11':
				if row['Label'] == '6' or row['Label'] == '9':
					sents.append({'sent':row['Sentence'], 'label':'7', 'category':row['Category']})
				else:
					sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA
	
def notdoing_vs_multiple_vs_other(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Leaves out all the sentenes except those with labels 11, 6, 7 and 9 (other). Used to classify sents within the category other as a4 not doing (11) vs. multiple (9) vs. other (6+7).
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] == '6' or row['Label'] == '7' or row['Label'] == '9' or row['Label'] == '11':
				if row['Label'] == '6':
					sents.append({'sent':row['Sentence'], 'label':'7', 'category':row['Category']})
				else:
					sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA

def multiple_vs_other(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Leaves out all the sentenes except those with labels 6, 7 and 9. Used to classify sents as multiple (9) vs. other (6+7).
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] == '6' or row['Label'] == '7' or row['Label'] == '9':
				if row['Label'] == '6':
					sents.append({'sent':row['Sentence'], 'label':'7', 'category':row['Category']})
				else:
					sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA

def notdoing_vs_multiple_vs_general_vs_other(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Leaves out all the sentenes except those with labels 11, 6, 7 and 9 (other). Used to classify sents within the category other as a4 not doing (11) vs. multiple (9) vs. general (6) vs. other (7).
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] == '6' or row['Label'] == '7' or row['Label'] == '9' or row['Label'] == '11':
				sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA	
	
def general_vs_other(file1):
	'''Reads in the file in the format: ID;Sentence;Label;Category.
	Leaves out all the sentenes except those with labels 6 and 7. Used to classify sents within the category other as general (6) vs. other (7) agendas.
    Returns: {ID:{'sent':'Sentence', 'label':'Label', 'category':'Category'}}'''
	import codecs
	import csv
	DATA = {}
	sents = []
	labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '11']
	with codecs.open(file1, 'r', 'utf-8') as raw:
		reader_raw = csv.DictReader(raw, delimiter='\t') #';' for TestLFCorpus
		for row in reader_raw:
			if row['Label'] == '6' or row['Label'] == '7':
				sents.append({'sent':row['Sentence'], 'label':row['Label'], 'category':row['Category']})
			if row['Category'] != 'jo' and row['Category'] != 'pm' and row['Category'] != 'sm' and row['Category'] != 'sc':
				print row['ID']
			if  row['Label'] not in labels:
				print row['Label']
		ids = range(len(sents)-1)
		for id in ids:
			DATA[id] = sents[id]
	return DATA	

def get_labels(data):
	'''Returns a list of labels retrieved from data dictionary produced by data_generator.''' 
	labels = []
	for i in data.keys():
		labels.append(data[i]['label'])
	return np.array(labels)

def get_a4a(file1, file2):
	'''
	This function generates input for fine-grain classification for non-labelled data.
	It takes two files as input:
	The first file is a result of 2-labelled classification in a form of <id><\t><lab>
	The second file is an original file with sentences.
	
	The result is a file that contains only those lines from the original file that are a4a saved to the same location as file2.
	@return: a path to the result file
	'''
	#path = file1[:file1.rfind('\\')]
	path = dirname(file1) + '\\'
	#print path
	#print path+basename(file2)[:basename(file2).rfind('.')]
	with codecs.open(file1, 'r', 'utf8') as labels:
		#print '1'
		positives = {line.strip().split('\t')[0]:line.strip().split('\t')[1] for line in labels.readlines() if line.strip().split('\t')[1] == 'dir'}
		#print '2'
	with codecs.open(file2, 'r', 'utf8') as sents:
		all_sents = sents.readlines()[1:]
		with codecs.open(path+basename(file2)[:basename(file2).rfind('.')]+'_positives.txt', 'w', 'utf8') as resfile:
			resfile.write('ID,Sentence,Label,Date\n')
			for sent in all_sents:
				for id in positives:
					#if sent.startswith(id+','):
					if sent.split(',')[0] == id:
						resfile.write(str(sent))
	print 'Got a4a'					
	return 	path+basename(file2)[:basename(file2).rfind('.')]+'_positives.txt'

def correct_id(data, suff):
	'''
	Takes DATA enerated by data_generator_xx() and corrects the id for not labelled data so that no missing ids.
	'''
	with codecs.open(r"C:\Users\stolpovskaya\Dropbox\thesis\\"+suff+'\id_mapping_positives.txt', 'w', 'utf8') as map:
		new_data = {}
		ids = range(len(data))
		#for id, item in zip(ids, data.values()):
		for id, item in zip(ids, data.keys()):
			new_data[id] = data[item]
			map.write(str(id+1) + '\t' + str(item) + '\n')
		#print len(new_data)
		#print str(new_data[1])
		return new_data	
	
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
		category_mapping = {'jo':0, 'pm':1, 'sm':2, 'sc':3}
		items = []
		for i in range(len(data)):
			if self.key == 'sent':
				items.append(data[i][self.key])
			elif self.key == 'features':
				items.append(OrderedDict(sorted(data[i][self.key].items())))
			elif self.key == 'category':
				try:
					items.append([category_mapping[data[i][self.key]]])
				except:
					print data[i]
			elif self.key == 'vector':
				items.append(tokenizer_split(data[i][self.key]))
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