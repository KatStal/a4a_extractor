#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This file contains all the lists of words for LinguisticFeaturesExtractor_2.py
'''

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

MODAL_VERBS = ["must", "should", "need", "ought"]
HINT_VERBS = ["demand", "request", "command", "urge", "implore", "plead", "insist", "hope","pray", "impose", "forbid","instruct",
              "refuse","pursue", "threaten", "require", "order", "encourage","discourage","warn","strive","provoke",
              "welcome","applause","incite","threat","call","ask","have","ultimate","stipulate","ultimatum","stipulation",
	      "insistence","dictate","pressure","clamour","clamor","menace","intimidate","browbeat","bully","pressurize",
	      "terrorize","frighten","scare","alarm","oblige","tell", "want", "suppose", "promise", "swear", "vow", "offer", "suggest",
             "plan", "intent",  "pledge", "guarantee", "engage", "coerce","convince", "prevent", "persuade"]
HINT_ADJ = ["pointless", "senseless", "futile", "hopeless", "fruitless", "useless", "needless", "in vain", "unavailing", "necessary", "unnecessary", "unacceptable", "imperative", "obligatory", "requisite", "compulsory", "mandatory"]
VERB_PLAN = ["can", "ca", "going"]
VERB_STAND = ["stand","withstand","endure","tolerate"]
HINT_NOUNS = ["priority", "answer", "way", "solution", "approach","strategy","action","measure","step"] 
ADJ_VITAL = ["vital","important","significant", "essential", "substantial", "principal", "salient"]
VERB_BLAME = ["blame", "condemn", "deplore", "decry", "denounce"]
VERB_STATE = ["be", "feel"]
CONJS = ["who", "whom", "whoes", "if", "whether"]
LET_VERBS = ["Let", "let","Do not", "Do n't", "do not", "do n't"]
VERB_FUTURE = ["will", "shall","wo"]
#CLAUSE_MARKERS = ["(S (VP (VB ", "(S (VP (VBP ", "(SBAR (VP (VB ", "(SBAR (VP (VBP "]
CLAUSE_MARKERS = ["(S  (VP", "(S(VP", "(S (VP"]
NOUN_PROMISE = ["promise", "word", "assurance", "vow", "pledge", "guarantee", "oath"]
VERB_GIVE = ["give", "make", "hold", "gave", "given", "givin", "made", "making", "held"]

