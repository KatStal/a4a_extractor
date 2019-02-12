
#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import codecs
import re
import sys


# setting utf-8 env for console
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)


def detect_SBAR(line):
    '''Detecting SBAR sets in a line and extracting their content as a list. The returned list is empty if no SBAR was found'''
    i = 0
    results = []
    while True:
        try:
            start_i = line.index(('SBAR'), i)
            results.append(parse_SBAR(line, start_i-1)) # start-1 because SBAR is always preceeded by a (
            i = start_i + 1
        except:
            return results

def detect_ROOT_S_1(line):
    '''Detecting SBAR sets in a line and extracting their content as a list. The returned list is empty if no SBAR was found'''
    i = 0
    results = []
    while True:
        try:
            start_i = line.index(('ROOT (S '), i)
            results.append(parse_SBAR(line, start_i-1)) # start-1 because SBAR is always preceeded by a (
            i = start_i + 1
        except:
            return results

def detect_ROOT_S(line):
    '''Detecting SBAR sets in a line and extracting their content as a list. The returned list is empty if no SBAR was found'''
    i = 0
    results = []
    while True:
        try:
            start_i = line.index(('ROOT(S '), i)
            results.append(parse_SBAR(line, start_i-1)) # start-1 because SBAR is always preceeded by a (
            i = start_i + 1
        except:
            #print results
            return results	

def detect_S_VP(line):
    '''Detecting SBAR sets in a line and extracting their content as a list. The returned list is empty if no SBAR was found'''
    i = 0
    results = []
    while True:
        try:
            start_i = line.index(('(S(VP '), i)
            results.append(parse_SBAR(line, start_i-1)) # start-1 because SBAR is always preceeded by a (
            i = start_i + 1
        except:
            #print results
            return results

def parse_SBAR(substr, i):
    '''Extracting SBAR text'''
    text = []
    br = 0
    index = 0
    for c in substr[i:]:
        if c == '(':
            br += 1
        if c == ')':
            br -= 1
        if br < 0:
            break
        else:
            text.append(c)
            index += 1

    full_text = ''.join(text)
    #print full_text # not necessary!
    return clean_text(full_text)


def clean_text(text):
    '''Cleaning up output text'''
    # remove markup: we assume that each label is immediately preceeded by a ( and consists of A-Z chars
    spaces_text = re.sub(r'\([A-Z]+|\)', '', text)
    # remove extra spaces
    clean_text = re.sub(r' +', ' ', spaces_text)
    clean_text = re.sub(r'^ | $', '', clean_text)
    #return (clean_text, '\n')
    return (clean_text)

if __name__ == "__main__":
    print 'sbar: ', detect_SBAR(u'(ROOT (S (NP (NN Conflict)) (VP (VBZ does) (RB not) (ADVP (RB necessarily)) (VP (VB have) (SBAR (VP (TO to) (VP (VB result) (PP (IN in) (NP (NN violence)))))))) (. .)))')
    print 'root_s: ', detect_ROOT_S_1(u'(ROOT (S (NP (NN Conflict)) (VP (VBZ does) (RB not) (ADVP (RB necessarily)) (VP (VB have) (SBAR (VP (TO to) (VP (VB result) (PP (IN in) (NP (NN violence)))))))) (. .)))')
    print 'root_s_1: ', detect_ROOT_S("(ROOT(S  (VP (VB Let) (S(NP (PRP 's))(VP (VB sing)  (PRT (RP along)))))  (. .)))")
    print 'sbar1: ', detect_SBAR(u'(ROOT\n  (S\n    (NP (PRP It))\n    (VP (VBZ is)\n      (ADJP (JJ hard)\n        (S\n          (VP (TO to)\n            (VP (VB say)\n              (SBAR (IN whether)\n                (S\n                  (NP (DT this))\n                  (VP (MD must)\n                    (VP (VB be)\n                      (VP (VBN done)))))))))))\n    (. .)))')
    print 's_vp: ', detect_S_VP("(ROOT(S  (NP (PRP They))  (VP (VBD said) (: :) (`` `) (S(VP (VB Let)  (S (NP (PRP us)) (VP (VB do)(NP (PRP it)))))))  (. .) ('' ')))")    
#print detect_ROOT_S_1(u'(ROOT (NP (NP (NNP WASHINGTON)) (: --) (S (S (NP (NP (DT The) (JJ Serbian) (NN province)) (PP (IN of) (NP (NNP Kosovo)))) (VP (VBZ is) (NP (NP (DT a) (NN tinderbox)) (SBAR (WHNP (WDT that)) (S (VP (VBZ poses) (N$


