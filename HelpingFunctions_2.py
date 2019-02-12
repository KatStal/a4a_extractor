
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Helping functions for LinguisticFeaturesExtractor_2.py
'''

from lists import *
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

###
### Helping functions
###

def check_tag_verb_inf(pos):
  if pos == 'MD' or pos == 'VB':
    return (True)

def check_tag_verb_any(pos):
  if pos == 'MD' or pos == 'VBG' or pos == 'VB' or pos == 'VBP' or pos == 'VBZ' or pos == 'VBD':
    return (True)

def check_tag_participle(pos):
  if pos == 'VBN':
    return (True)

#def check_tag_noun(pos, data_nlp):
#  for i in range(len(data_nlp[0]['tokens'])):
#    if data_nlp[0]['tokens'][i]['pos'].startswith('NN'):
#      return (True)

def check_tag_noun(pos):
  if pos.startswith('NN'):
      return (True)

def check_tag_uh(pos):
  if pos.startswith('UH'):
      return (True)

def check_verb_tense_past(pos):
  if pos == 'VBD':
    return (True)

def check_verb_tense_future(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    for verb in VERB_FUTURE:
      if "aux" in item.values() and word in item.values() and verb in item.values():
        return (True)

def check_rel_aux(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    #print item
    for verb in VERB_STATE:
      if "aux" in item.values() and word in item.values() and verb in item.values():
        return (True)

def check_infinitive(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    if "mark" in item.values() and word in item.values() and "to" in item.values():
      return (True)

def check_rel_amod(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    if "amod" in item.values() and word in item.values():
      return (True)

def check_rel_vmod(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    if ("vmod" in item.values() or "acl" in item.values()) and word in item.values():
      return (True)

def check_rel_prepclause(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    #print item.values()
    #print word
    if ("auxpass" in item.values() or "nmod:for" in item.values() or "nmod:upon" in item.values() or "nmod:from" in item.values()) and word in item.values():
      return (True)

def check_rel_clause(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    #print item.values()
    if ("xcomp" in item.values() or "ccomp" in item.values()) and word in item.values():
       return (True)

def check_rel_cop(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    if "cop" in item.values() and word in item.values():
       return (True)

def check_rel_prep_as(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    if ("nmod:as" in item.values() or "advcl:as" in item.values()) and word in item.values():
       return (True)

def check_rel_plan_stand(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    for verb in VERB_PLAN:
      if ("xcomp" in item.values() or "aux" in item.values()) and word in item.values() and verb in item.values():
        return(True)

def check_rel_neg(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    if "neg" in item.values() and word in item.values():
      return (True)

def check_rel_dobj(word, data_nlp):
  for item in data_nlp[0]['collapsed-dependencies']:
    if "dobj" in item.values() and (word == item['dependentGloss'] or word == item['governorGloss']):
      return (True)
