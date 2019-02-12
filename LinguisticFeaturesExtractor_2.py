#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
To test this script run TestLFCorpus.txt.
Run with no virtual environment activated.
Run as sudo user.
Pass the name of the file you would like to extract linguistic features from as an argument.
'''

#from corenlp import *
import sys
import csv
from SBAR_parser import *
import codecs
from HelpingFunctions_2 import *
from lists import *
import ast
import json
#import simplejson as json

#corenlp = StanfordCoreNLP()
from pycorenlp import StanfordCoreNLP
corenlp = StanfordCoreNLP('http://localhost:9000')

reload(sys)
sys.setdefaultencoding('utf-8')

###
### Core function
###
	  
def FeatureExtractor (data_nlp, id): 
	features = {}
	# Feature 1: if a MODAL_VERBS is in the sent
	# Feature 2: if a MODAL_VERBS is tagged as a MD (modal auxiliary) or VB in the sent	
	# Feature 3: if a MODAL_VERBS is in aux relationship with "be"/"feel"
	# Feature 4: if a MODAL_VERBS is followed by "have" + VBN

	for i in range(len(data_nlp[0]['tokens'])):
		for word in MODAL_VERBS:
			if word == data_nlp[0]['tokens'][i]['lemma']:
				features["Modal verb is in the sent"]="1"

          			if check_tag_verb_any(data_nlp[0]['tokens'][i]['pos']):
	        			features["Modal verb has MD or VB tag"]="1"
			
		        	if check_rel_aux(data_nlp[0]['tokens'][i]['word'], data_nlp):
			        	features["Modal verb is in aux rel with 'be'/'feel'"]="1"
			
			        if check_rel_prepclause(data_nlp[0]['tokens'][i]['word'], data_nlp) or check_rel_clause(data_nlp[0]['tokens'][i]['word'], data_nlp):
			        	features["Modal verb has a clause or prepositional phrase"]="1"
			  
			        try:
				    if data_nlp[0]['tokens'][i+1]['lemma'] == "have" and data_nlp[0]['tokens'][i+2]['pos'] == 'VBN':
		  			features["Modal verb is followed by 'have' + VBN"]="1"
			        except:
				    continue
                                
                                for l in range(len(detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ' ')))):
        		  	        for conj in CONJS:
        	        	    	    #print word in detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ' '))[l]
                         	            if word in detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ' '))[l] and conj in detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ' '))[l]:
        		      			features["Modal verb is a part of if/who clause"]="1"
			  
	# Feature 5: if a HINT_VERBS is in the sent
	# Feature 6: if a HINT_VERBS is tagged as a VB/VBG/VBP/VBZ
	# Feature 7: if a HINT_VERBS is tagged as VBD (past tense)
	# Feature 8-9: if a MODAL_VERBS OR HINT_VERBS is in auxpass OR xcomp OR ccomp OR prep_for OR prep_upon OR prepc_from relationship
	# Feature 10: if a HINT_VERBS is tagged as VBN (participle)
	# Feature 11-12: if a MODAL_VERBS or HINT_VERBS is a part of whether/if/who/whom clause (SBAR starting with whether/if/who/whom)

		for word in HINT_VERBS:
			if word == data_nlp[0]['tokens'][i]['lemma']:
				features["Hint verb is in the sent"]="1"
			
         			if check_tag_verb_any(data_nlp[0]['tokens'][i]['pos']):
			 	    features["Hint verb has a VB* tag"]="1"
			
			#if check_verb_tense_past(data_nlp[0]['tokens'][i]['pos']): # TO DO: why not combining with feature 6
			#	features["Hint verb is in past tense"]="1"
                                #print data_nlp[0]['tokens'][i]['word']			  
			        if check_rel_clause(data_nlp[0]['tokens'][i]['word'], data_nlp) or check_rel_prepclause(data_nlp[0]['tokens'][i]['word'], data_nlp):
                                    #print data_nlp[0]['tokens'][i]
				    features["Hint verb has a clause or prepositional phrase"]="1"
			  
			        if check_tag_participle(data_nlp[0]['tokens'][i]['pos']):
				    features["Hint verb is a past participle"]="1"		  

                                for q in range(len(detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ' ')))):
			            for conj in CONJS:
				        if word in detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ' '))[q] and conj in detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ' '))[q]:
        				    	features["Hint verb is a part of if/who clause"]="1" 

	# Feature 13: if "time" is in the sent
	# Feature 14: if "time" is in vmod relationship
	# Feature 15: if "time" is in amod relationship

		if data_nlp[0]['tokens'][i]['lemma'] == "time" and check_tag_noun(data_nlp[0]['tokens'][i]['pos']):
			features["Noun 'time' is in the sent"]="1" 
		
        		if check_rel_amod('time', data_nlp):
	        		features["Noun 'time' is modified by an adjective"]="1" # TODO: restrict the adj, "it is the right time to do it" is a proper agenda

		        if check_rel_vmod('time', data_nlp):
			        features["Noun 'time' is modified by a verbal modifier"]="1"

	# Feature 16: if "please" is in the sent

		if data_nlp[0]['tokens'][i]['lemma'] == "please" and check_tag_uh(data_nlp[0]['tokens'][i]['pos']): 
			features["The sent contains 'please'"]="1"

	# Feature 17: if HINT_ADJ is in the sentence
	# Feature 18: if HINT_ADJ is in cop OR prep_as relationship

		for word in HINT_ADJ:
			if word == data_nlp[0]['tokens'][i]['lemma']:
				features["Hint adjective is in the sent"]="1"
			
         			if check_rel_cop(data_nlp[0]['tokens'][i]['word'], data_nlp):
	        			features["Hint adjective is a part of the predicate"]="1"
			
		        	if check_rel_prep_as(data_nlp[0]['tokens'][i]['word'], data_nlp):
			        	features["Hint adjective is a part of the prepositional phrase"]="1"

	# Feature 19: if VERB_PLAN is in xcomp or aux relationship with VERB_STAND
	# Feature 20: if VERB_PLAN is in neg relationship    
	# Feature 21: if VERB_STAND is in aux relationship with "will"
	# Feature 22: if VERB_STAND is in neg relationship

		for word in VERB_STAND:
			if word == data_nlp[0]['tokens'][i]['word']:
				if check_rel_plan_stand(data_nlp[0]['tokens'][i]['lemma'], data_nlp):
					features["'Can/can't/going (to) stand' is in the sent"]="1"
			
		                if check_verb_tense_future(data_nlp[0]['tokens'][i]['word'], data_nlp):
                                    features["The verb meaning 'stand' is in future tense"]="1"

			        if check_rel_neg(data_nlp[0]['tokens'][i]['word'], data_nlp):
			    	    features["The verb meaning 'stand' is negated"]="1"

		for word in VERB_PLAN:
			if word == data_nlp[0]['tokens'][i]['word'].lower():
				if check_rel_neg(data_nlp[0]['tokens'][i]['word'], data_nlp):
					features["'Can/can't/going' is negated"]="1"

	# Feature 23: if HINT_NOUNS is in the sentence
	# Feature 24: if HINT_NOUN is tagged as NN or NNS
	# Feature 25-27: if HINT_NOUNS is in prep_as OR cop relationship, or is a direct object

		for word in HINT_NOUNS:
			if word == data_nlp[0]['tokens'][i]['lemma']:
				features["Hint noun is in the sent"]="1"

			        if check_tag_noun(data_nlp[0]['tokens'][i]['pos']):
			    	    features["Hint noun is tagged as a noun"]="1"
			  
			        if check_rel_dobj(data_nlp[0]['tokens'][i]['word'], data_nlp):
			   	    features["Hint noun is a direct object"]="1"

			        if check_rel_prep_as(data_nlp[0]['tokens'][i]['word'], data_nlp):
			    	    features["Hint noun is a part of the prepositional phrase"]="1"

			        if check_rel_cop(data_nlp[0]['tokens'][i]['word'], data_nlp):
			    	    features["Hint noun is a part of the predicate"]="1"

	# Feature 28: if ADJ_VITAL is in ccomp relationship

		for word in ADJ_VITAL:
			if word == data_nlp[0]['tokens'][i]['lemma'] and check_rel_clause(data_nlp[0]['tokens'][i]['word'], data_nlp):
				features["The sentence says 'It is important to do something"]="1"
	 
	# Feature 38: if  NOUN_PROMISE is in the sentence and is a direct object of a VERB_GIVE
                for word in NOUN_PROMISE:
                        for verb in VERB_GIVE:
                            if word == data_nlp[0]['tokens'][i]['lemma']:
                                if check_rel_dobj(word, data_nlp) and check_rel_dobj(verb, data_nlp):
                                    features['The sentence has an expression "give a promise" or similar']="1"

        # Feature 29: if VERB_BLAME is in the sentence
	# Feature 30: if VERB_BLAME is tagged as VB/VBG/VBP/VBZ
	# Feature 31: if VERB_BLAME is in aux relation with "to"

		for word in VERB_BLAME:
			if word == data_nlp[0]['tokens'][i]['lemma']:
				features["A word meaning 'blame' is in the sent"]="1"

                		if check_tag_verb_any(data_nlp[0]['tokens'][i]['pos']):
		        	#print 'Feature 30 found in sentence number', id, '\n'
			            features["A word meaning 'blame' is tagged as a verb"]="1"

		                if check_infinitive(data_nlp[0]['tokens'][i]['word'], data_nlp):
			        #print 'Feature 31 found in sentence number', id, '\n'
			            features["Blame is a part of a clause"]="1"

	# Feature 32: if there is a xcomp rel with verb "to be"

		if data_nlp[0]['tokens'][i]['lemma'] == "be":
			if check_rel_clause(data_nlp[0]['tokens'][i]['word'], data_nlp):
				#print 'Feature 32 found in sentence number', id, '\n'
				features["Verb 'to be' has an open clausal complement"]="1"

	# Feature 33: if the sentence ends with "?"
	# Feature 34: if the sentence ends with "!"
	# Feature 35: if the sentence (S or SBAR) starts with "don't" OR "do not" OR "let" OR "let's" 
	# Feature 36: if the sentence (S or SBAR) starts with a VB/VBP
	
		if "!" in data_nlp[0]['tokens'][i].values():
		#print 'Feature 34 found in sentence number', id, '\n'
			features["It is an exclamatory sentence"]="1"

		if "?" in data_nlp[0]['tokens'][i].values():
		#print 'Feature 33 found in sentence number', id, '\n'
			features["It is an interrogative sentence"]="1"

	for word in LET_VERBS:  # TO DO: redundunt feature, as "let", "do" are also tagged as verbs and they start a sentence/claus.
            #print 'give this to root_s: ', data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")
            #print detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ""))
            if len(detect_ROOT_S(data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")))>0:
                for m in range(len(detect_ROOT_S(data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")))):
                    if detect_ROOT_S(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ""))[m].lower().startswith(word):#  and features["It is an imperative sentence/clause starting with 'don't'/'let'"] != "1":
                        features["It is an imperative sentence/clause starting with 'don't'/'let'"]="1"
            if len(detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")))>0:
                for j in range(len(detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")))):
                     if detect_SBAR(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ' '))[j].lower().startswith(word):# and features["It is an imperative sentence/clause starting with 'don't'/'let'"] != "1":
                         features["It is an imperative sentence/clause starting with 'don't'/'let'"]="1"
            if len(detect_S_VP(data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")))>0:
                #print 's_vp: ', detect_S_VP(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ""))
                #print 'len s_vp: ', len(detect_S_VP(data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")))
                for y in range(len(detect_S_VP(data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")))):
                     #print 'y: ', y
                     #print detect_S_VP(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ""))[y].lower().startswith(word)
                     if detect_S_VP(data_nlp[0]['parse'].replace('\n', " ").replace('   ', ""))[y].lower().startswith(word):# and features["It is an imperative sentence/clause starting with 'don't'/'let'"] != "1":
                         features["It is an imperative sentence/clause starting with 'don't'/'let'"]="1"

	
        
        for marker in CLAUSE_MARKERS:
	    #print marker
            #print data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")
            #print marker in data_nlp[0]['parse'].replace('\n', " ").replace('   ', "")	
            if marker in data_nlp[0]['parse'].replace('\n', " ").replace('   ', ""):
			#print 'Feature 36 found in sentence number', id, '\n'
		  	features["The sent/clause begins with a verb"]="1"
	return features
	
features = {}
infile = sys.argv[1]
with codecs.open(infile, 'r', 'utf-8') as csvfile:
	#reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
	#for row in reader:
	#	print '\n', 'ID: ', row['ID']
        #        print row['Sentence']
        #        print row['Label']
	lines = csvfile.readlines()[1:]
	for line in lines:
		id = line[:line.find(',')]
		sent = str(line[line.find(',')+1:line.find(',???,')])
		print '\n', 'ID: ', id
		print sent
		#print corenlp.parse(row['Sentence']).values()	
		#features[row['ID']]=FeatureExtractor(corenlp.parse(row['Sentence'])['sentences'], row['ID'])
		nlp_data = corenlp.annotate(sent, properties={'annotators': 'tokenize,lemma,pos,parse','outputFormat': 'json'})
		print type(nlp_data)
		if isinstance(nlp_data, unicode):
			#nlp_data = ast.literal_eval(nlp_data)
			nlp_data = json.loads(nlp_data, strict=False)
		features[id]=FeatureExtractor(nlp_data['sentences'], id)	  

print "You have", len(features), "data items. \n\n"
print '..........................................................................\n\n'
	
LIST_FEATURES = ["Modal verb is in the sent","Modal verb has MD or VB tag","Modal verb is in aux rel with 'be'/'feel'","Modal verb has a clause or prepositional phrase",
                 "Modal verb is followed by 'have' + VBN","Modal verb is a part of if/who clause","Hint verb is in the sent","Hint verb has a VB* tag",
                 "Hint verb has a clause or prepositional phrase","Hint verb is a past participle","Hint verb is a part of if/who clause" ,"Noun 'time' is in the sent",
                 "Noun 'time' is modified by an adjective","Noun 'time' is modified by a verbal modifier","The sent contains 'please'","Hint adjective is in the sent",
                 "Hint adjective is a part of the predicate","Hint adjective is a part of the prepositional phrase","'Can/can't/going (to) stand' is in the sent","Hint noun is a direct object",
                 "The verb meaning 'stand' is in future tense","The verb meaning 'stand' is negated","'Can/can't/going' is negated","Hint noun is in the sent","Verb 'to be' has an open clausal complement",
                 "Hint noun is tagged as a noun","Hint noun is a part of the prepositional phrase","Hint noun is a part of the predicate","The sentence says 'It is important to do something'",
                 "A word meaning 'blame' is in the sent","A word meaning 'blame' is tagged as a verb","Blame is a part of a clause","It is an exclamatory sentence",
                 "It is an interrogative sentence","It is an imperative sentence/clause starting with 'don't'/'let'","The sent/clause begins with a verb",'The sentence has an expression "give a promise" or similar']

for id in features.keys():
  for feature in LIST_FEATURES:
    if len(features[id])==0 or feature not in features[id].keys():
      features[id][feature]='0'

#print features['1']	  
print '..........................................................................\n\n'
print "Features have been successfully extracted!\n"    

#resfile = r'/home/amcat/AgendasForAction/errors_LF.csv'
#resfile = r'/home/amcat/AgendasForAction/corpus_5000_LF.csv'
#resfile = r'/home/amcat/AgendasForAction/data_with_features_test.csv'
#resfile = r'/home/amcat/AgendasForAction/dir_vs_ndir_big_train_corpus_2.csv'
#resfile = r'/home/amcat/AgendasForAction/Test_LF.csv'
#resfile = r'/home/amcat/AgendasForAction/corpus_15K_allLab_LF.csv'
resfile = r'/home/amcat/AgendasForAction/NYT_LF5_new.csv'
with codecs.open(resfile, 'w', 'utf-8') as csvfile:
	csvfile.write('ID;Feature;value\n')
	for id in features:
		for feat, val in features[id].items():
			csvfile.write(str(id) + ';' +str(feat)+';'+str(val)+'\n')		

print 'Results have been saved to ', resfile, '\n'

###############
### Testing ###
###############
if "TestLFCorpus.txt" in infile:
        with codecs.open(r'/home/amcat/AgendasForAction/TestLF_gold.txt', 'r', 'utf8') as gold:
		correct = gold.readlines()[1:]
		with codecs.open(resfile, 'r', 'utf8') as results:
			real = results.readlines()[1:]
			for line in real:
				if not line in correct:
					print "LF are not extracted in accordance with golden standart!"
					print line
