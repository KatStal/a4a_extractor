#!/usr/bin/python
# -*- coding: utf-8 -*-

### Extracts a4a based on concepts coded

###
### Run as: python a4a_extractor_1.py <project_id> <set_no> <lang>
### If you extract a4a from multilingual set, use INDEX as the third argument, e.g., python a4a_extractor.py 15 451 INDEX
###

import re
import sys
import codecs
from nlp_preproc import tokenizer_split, split_into_sentences
from collections import defaultdict

projectid = sys.argv[1]
setid = sys.argv[2]
dictionary=sys.argv[3]
reload(sys)
sys.setdefaultencoding('utf-8')
mapping = {630:493, 627:497, 628:495, 629:494, 631:490, 632:492, 633:518, 636:257, 640:216, 637:217, 634:260, 635:259, 641:533, 617:293, 615:296, 616:294, 614:297, 613:298, 618:295, 620:331, 643:450, 639:452, 625:456, 645:451, 638:453, 644:455, 605:265, 604:266, 603:267, 606:264, 602:268, 607:269, 650:271, 649:274, 651:270, 648:273, 647:272, 619:275} 
	
def get_agendas_res(): 
	if dictionary=='INDEX':
		print 'looking up the INDEX file ...'
		#with codecs.open(r'E:\Soft\Python\Summer School\lang_index_'+str(projectid)+'_'+str(setid)+'.csv', 'r', 'utf8') as langs:
		with codecs.open(r'lang_index_'+str(projectid)+'_'+str(setid)+'.csv', 'r', 'utf8') as langs:
			lg_in = {line.split(',')[0]:line.split(',')[1].strip() for line in langs.readlines()}
			
	lang = dictionary
	check_lang = False
		
	concepts_en =[ '(Persuade/Persuasion)','(Warn/Warning)','(Demand/Urge NOT as in supply/demand)','(Praise/Applaud/Compliment//Hail)','(Request/Ask/Call Upon)','(Discourage)','(Encourage)','(Invite/Welcome)','(Threaten/Threat/Intimidate)','(Commit To/Pledge/Promise NOT Promised Land)','(Criticize/Condemn/Verbally Attack)','(Impose/Dictate)','(Order/Command)','(Pressure)','(Urgent)','(Unacceptable/Inappropriate)','(Unnecessary/Avoidable)',"(Agenda/Intention (someone's))",'(Needs/Requirements/Necessities)'] #'(Advice)','(Suggest/Indicate)','(Promote/Raise Support/Advocate)',
	words_en = [' must ', ' should ', ' ought ', ' have to ', ' had to ', ' has to ', 'require ', 'requires'] #'required' and 'requiering' were not included as they are used as modifiers often
	no_words_en = [' in order ']
	concepts_de =[ '(Persuade/Persuasion)','(Warn/Warning)','(Demand/Urge NOT as in supply/demand)','(Praise/Applaud/Compliment//Hail)','(Request/Ask/Call Upon)','(Discourage)','(Encourage)','(Invite/Welcome)','(Commit To/Pledge/Promise NOT Promised Land)','(Criticize/Condemn/Verbally Attack)','(Impose/Dictate)','(Order/Command)','(Pressure)','(Urgent)','(Unacceptable/Inappropriate)','(Unnecessary/Avoidable)',"(Agenda/Intention (someone's))",'(Promote/Raise Support/Advocate)'] #'(Advice)','(Suggest/Indicate)','(Promote/Raise Support/Advocate)', ,'(Needs/Requirements/Necessities)','(Threaten/Threat/Intimidate)'
	words_de = ['müsse', 'müsst', 'muss ', 'musse', 'musst', 'soll ', 'sollst', 'solle', 'sollt']
	concepts_fr =[ '(Persuade/Persuasion)','(Warn/Warning)','(Demand/Urge NOT as in supply/demand)','(Praise/Applaud/Compliment//Hail)','(Request/Ask/Call Upon)','(Discourage)','(Encourage)','(Criticize/Condemn/Verbally Attack)','(Impose/Dictate)','(Order/Command)','(Pressure)','(Urgent)','(Unacceptable/Inappropriate)','(Unnecessary/Avoidable)',"(Agenda/Intention (someone's))",'(Threaten/Threat/Intimidate)','(Promote/Raise Support/Advocate)'] #'(Advice)','(Suggest/Indicate)', ,'(Needs/Requirements/Necessities)','(Invite/Welcome)','(Commit To/Pledge/Promise NOT Promised Land)'
	words_fr = []
	concepts_all =['(Persuade/Persuasion)','(Warn/Warning)','(Demand/Urge NOT as in supply/demand)','(Praise/Applaud/Compliment//Hail)','(Request/Ask/Call Upon)','(Discourage)','(Encourage)','(Criticize/Condemn/Verbally Attack)','(Impose/Dictate)','(Order/Command)','(Pressure)','(Urgent)','(Unacceptable/Inappropriate)','(Unnecessary/Avoidable)',"(Agenda/Intention (someone's))",'(Threaten/Threat/Intimidate)','(Promote/Raise Support/Advocate)','(Advice)','(Suggest/Indicate)', '(Needs/Requirements/Necessities)','(Invite/Welcome)','(Commit To/Pledge/Promise NOT Promised Land)'] #
	words_all = []

	concept_deesc = []
	concept_esc = []
	concept_pun = []
	concept_help = []
	concept_igno = []

	res = defaultdict(list)
	sents = defaultdict(list)
	negatives = defaultdict(list)
	with codecs.open(r'annotated_'+str(projectid)+'_'+str(setid)+'.txt', 'r', 'utf8') as input:
		arts = input.read().split('\n\n') # a list of annotated articles, each of them is a string
		arts_split = [art.replace('Etc.)','Etc)').replace('incl. ', 'incl ').split('\t') for art in arts] # a nested list [[id, headline, text]]
			
		for i in arts_split:
			if len(i) == 3:
				sents[i[0]] += ([i[1]]+split_into_sentences(i[2]))
			elif len(i)== 4:
				sents[i[0]] += ([i[1]]+split_into_sentences(i[2])+split_into_sentences(i[3]))
		
		for id, sent in sents.items():
			if lang == 'INDEX' or check_lang:
				check_lang = True
				lang = lg_in[id]
			for s in sent:
				if lang == 'EN':
					#print "Extracting a4a for EN",
					for w in (concepts_en+words_en):
						for nw in no_words_en:
							#if w in s: res[id].append({s:''}) # re.sub(r'\([^)]*\)', '', filename)   OR re.sub(r'\(.*?\)', '', s)
							if w in s and nw not in s and s not in res[id]: res[id].append(s) 
							else: negatives[id].append(s)

				elif lang == 'DE':
					#print "Extracting a4a for DE",
					for w in (concepts_de+words_de):
						#if w in s: res[id].append({s:''}) # re.sub(r'\([^)]*\)', '', filename)   OR re.sub(r'\(.*?\)', '', s)	
						if w in s and s not in res[id]: res[id].append(s) 
						else: negatives[id].append(s)
		
				elif lang == 'FR':
					#print "Extracting a4a for FR"
					for w in (concepts_fr+words_fr):
						#if w in s: res[id].append({s:''}) # re.sub(r'\([^)]*\)', '', filename)   OR re.sub(r'\(.*?\)', '', s)
						if w in s and s not in res[id]: res[id].append(s) 
						else: negatives[id].append(s)
			
				else:
					#print "Extracting a4a for all"
					for w in (concepts_all+words_all):
						#if w in s: res[id].append({s:''}) # re.sub(r'\([^)]*\)', '', filename)   OR re.sub(r'\(.*?\)', '', s)
						if w in s and s not in res[id]: res[id].append(s) 
						else: negatives[id].append(s)

	print 'I found ' + str(len(res)) + ' agendas for action'
	return res, set(negatives)
	
def get_texts_extra(project,set,user,pw,result_dict):
	from amcatclient import AmcatAPI
	from elasticsearch import Elasticsearch
	import string
	texts=[]
	conn=AmcatAPI('http://jamcat.mscc.huji.ac.il',user,pw)
	#es = Elasticsearch(['http://jamcat.mscc.huji.ac.il:9200'])
	es = Elasticsearch()
	for id, b in result_dict.items():
		result = es.search(index="amcat", doc_type="article", body = {"query": {"match": {"id":id}}})
		try:
			r=result['hits']['hits'][0]['_source']
			medium=r['medium']
			date=r['date']
			title=''
			subtitle=''
			letters = string.ascii_lowercase[0:len(b)+1]
			for i in range(len(b)):
				text=re.sub(r'\(.*?\)', '', b[i])
				id_ = str(id)+'_'+str(letters[i])
				article=[id_,medium,date,title,subtitle,text]
				texts.append(article)
		except:
			print str(id) + ' is not in the database anymore'	
	print len(texts),'texts retrieved...'
	return texts

def count_words_char(sent_list):
	words = 0
	text = ''
	for sent in sent_list:
		words += len(tokenizer_split(sent)) 
		text += sent.replace(' ', '')
	return words, len(text)

if __name__ == "__main__":
	from elasticsearch import Elasticsearch
	es = Elasticsearch()
	pos, neg = get_agendas_res()
	with codecs.open(r'found_a4a_'+str(projectid)+'_'+str(setid)+'.txt', 'w', 'utf8') as output:
		for a, b in pos.items():
			output.write('\n'+ str(a) + '\n')
			for a4a in b:
				output.write(str(a4a) + '\n')
				
		print 'Agendas for action from ' + str(projectid) + '-' + str(setid) + ' have been written to ' + 'found_a4a_'+str(projectid)+'_'+str(setid)+'.txt' + '.' 		

	###############################################################################
	### UNCOMMENT HERE TO GENERATE THE FILE WITH NON-A4A (USEFUL FOR DEBUGGING) ###
	###############################################################################
	
	#with codecs.open(r'found_negative_a4a_'+str(projectid)+'_'+str(setid)+'.txt', 'w', 'utf8') as output:
	#	for a, b in neg.items():
	#		output.write('\n'+ str(a) + '\n')
	#		for a4a in b:
	#			output.write(str(a4a) + '\n')

	####################################################################################################
	### UNCOMMENT HERE TO GENERATE THE FILE WITH the number of words and characters in a4a sentences ###
	####################################################################################################
	
	# with codecs.open(r'found_a4a_stat_'+str(projectid)+'_'+str(setid)+'.txt', 'w', 'utf8') as output:
		# output.write('text_id' + '\t' + 'number of a4a' + '\t' + 'number of words in a4a sentences per document' + '\t' + 'number of characters in a4a sentences per document' + '\t' + 'length in words of the document' + '\t' + 'length in characters of the document' +'\n')
		# for a, b in get_agendas_res().items():
			# result = es.search(index="amcat", doc_type="article", body = {"query": {"match": {"id":a}}})
			# words, char = count_words_char(b)
			# output.write(str(a) + '\t' + str(len(b)) + '\t' + str(words) + '\t' + str(char) + '\t' + str(result['hits']['hits'][0]['_source']['length']) + '\t' + str(len(result['hits']['hits'][0]['_source']['text'].replace(' ', '')) + len(result['hits']['hits'][0]['_source']['byline'].replace(' ', '')) + len(result['hits']['hits'][0]['_source']['headline'].replace(' ', ''))) + '\n') #+ '\t' + str(len(result['hits']['hits'][0]['_source']['text'].replace(' ', '')) + len(result['hits']['hits'][0]['_source']['byline'].replace(' ', '')) + len(result['hits']['hits'][0]['_source']['headline'].replace(' ', '')))
		
		# print 'Stats about agendas for action from ' + str(projectid) + '-' + str(setid) + ' have been written to ' + 'found_a4a_stat_'+str(projectid)+'_'+str(setid)+'.txt' + '.' 

