# -*- coding: utf-8 -*-
""" This script does basic NLP pre-processing: word-tokenizing, sent-tokenizing, lemmatizing, pos-tagging"""
 
import nltk
import sys
from string import punctuation
import re
from nltk.stem.snowball import GermanStemmer

reload(sys)
sys.setdefaultencoding('utf-8')

#pre-processing tools
sents_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#sents_tokenizer_de = nltk.data.load('tokenizers/punkt/german.pickle')
stemmerEn = nltk.PorterStemmer() # uses nltk Porter stemmer
wnl = nltk.WordNetLemmatizer()
stemmerDe = GermanStemmer() # uses nltk Snowballs stemmer for German

def split_into_sentences(text):
	import re
	caps = "([A-Z])"
	prefixes = "(Mr|St|Mrs|Ms|Dr|dr|etc|vs|doc|art|no|inc|mr)[.]"
	suffixes = "(Inc|Ltd|Jr|Sr|Co|gdp|hon)"
	starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
	acronyms = "([A-Za-z][.][A-Za-z][.](?:[A-Za-z][.])?)"
	websites = "[.](com|net|org|io|gov|de|fr|il|mk)"
	dates = "(\d\d?)\.(\s+(januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember|jahrestag))"
	#dates = "(\d\d?)\."
	www = "(www)\."
	times = "(\d\d?)\.(\s?\d\d?)"
	full_date ="(\d\d?)\.(\s?\d\d?)\.(\s?\d\d\d?\d?)"
	
	text = " " + text + "  "
	text = text.replace("\n"," ")
	text = text.replace("\r"," ")
	text = re.sub(prefixes,"\\1<prd>",text)
	text = re.sub(websites,"<prd>\\1",text)
	text = re.sub(www,"\\1<prd>",text)
	text = re.sub(full_date,"\\1<prd>\\2<prd>\\3",text)
	text = re.sub(dates,"\\1<prd>\\2",text)
	#text = re.sub(dates,"\\1<prd>",text)
	text = re.sub(times,"\\1<prd>\\2",text)
	if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
	if "m.sci." in text: text = text.replace("m.sci.", "m<prd>sci<prd>")
	if "e.g." in text: text = text.replace("e.g.","e<prd>g<prd>")
	if "e.v." in text: text = text.replace("e.v.","e<prd>v<prd>")
	if "z.b." in text: text = text.replace("z.b.","z<prd>b<prd>")
	if "h.w." in text: text = text.replace("h.w.","h<prd>w<prd>")
	if "i.e." in text: text = text.replace("i.e.","i<prd>e<prd>")
	if 	"et al." in text: text = text.replace("et al.","et al<prd>")
	if 	"u. a." in text: text = text.replace("u. a.","u<prd>a<prd>")
	if 	"a. d." in text: text = text.replace("a. d.","a<prd>d<prd>")
	#if "vs." in text: text = text.replace("vs.","vs<prd>")
	text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
	text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
	text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
	text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
	text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
	text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
	text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
	if "”" in text: text = text.replace(".”","”.")
	if "\"" in text: text = text.replace(".\"","\".")
	if "!" in text: text = text.replace("!\"","\"!")
	if "?" in text: text = text.replace("?\"","\"?")
	text = text.replace(".",".<stop>")
	text = text.replace("?","?<stop>")
	text = text.replace("!","!<stop>")
	text = text.replace("<prd>",".")
	sentences = text.split("<stop>")
	sentences = sentences[:-1]
	sentences = [s.strip() for s in sentences]
	return sentences
	
def sent_tokenizer (text):    #takes raw text as input, returns a list of sents
	return (sents_tokenizer.tokenize(text.replace('/n', ' '))) 

def tokenizer_nltk(text): #takes raw text as input, returns a list of lower cased tokens
    return ([word.lower() for word in nltk.word_tokenize(text)])    

def tokenizer_split(text): #takes raw text as input, returns a list of lower cased tokens, doesn't count punctuation
    return ([word.lower() for word in re.split('[%s]+|\s' %punctuation, text) if word != ''])
	
def stemmer_en(text):  #takes raw text as input, returns a list of stems
    return ([stemmerEn.stem(word) for word in tokenizer_split(text)])

def stemmer_de(text):  #takes raw text as input, returns a list of stems
    return ([stemmerDe.stem(word) for word in tokenizer_split(text)])
    
def lemmatizer(text):  #takes raw text as input, returns a list of lemmata
    return ([wnl.lemmatize(word) for word in tokenizer_split(text)])

def lemmatizer_string(text):  #takes raw text as input, returns a string of lemmata
	return (' '.join(lemmatizer(text))) 
	
def tagger(text): #takes raw text as input, returns a list of tuples (word, POS)
    return (nltk.pos_tag(tokenizer_split(text)))	
	
	
if __name__ == '__main__':
    f = open(r'E:\Thesis\ACL\Texts_raw\401-500.txt', 'r')
    raw = f.read().decode("utf-8")
    text = raw.replace("\n", " ")
    #text = sys.stdin.read()
    text_sents = sent_tokenizer(text)
    text_tokens_nltk = tokenizer_nltk(text)
    text_tokens_split = tokenizer_split(text)
    text_stems_en = stemmer_en(text)
    text_stems_de = stemmer_de(text)
    text_lemmat = lemmatizer(text)
    text_tagged = tagger(text)
	
    output = 'preproc_output.txt'
    with open(output, 'w') as f:
#        for i in (text_sents, text_tokens, text_stems, text_tagged):
        for i in (text_sents):
            f.write("{}\n".format(str(i.encode("utf-8"))))
    print "Done pre-processing!"
    print "Output saved into", output 