# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:45:46 2017

@author: Ryan Compton
"""

import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.util import ngrams
from collections import Counter
import numpy as np
import csv
import sys

class SLM:
    #Inputs:
    #   text is a string holding the total amount of text from the sample
    #   tools is a list of tools to account for multiple text sources:
    #       Expected: "Wiki", "Blog", "Forum", "BlogCmt", "ForumRep"
    def __init__(self, text,tools=None,encoding="utf-8"):
        #Store the text provided
        self.all_text = text
        #Store the tool information for filtering which tool to use
        self.tools = tools
        #Set encoding to "utf-8" but allow for other encodings
        #Another one to expect is "latin-1"
        self.encoding = encoding
        #Initialize lemmatize model
        self.lemma = nltk.wordnet.WordNetLemmatizer()
        #Initialze total tri-gram counter dictionary
        self.tri_counts = Counter()
        #Build the Statistical Language Model
        print("Building SLM")
        self.build_model()
        print("Build Complete")
        
    #Need to separate out all text by sentences and then do tri-gram counts
    def build_model(self):
        #Gather the sentences within the text
        sentences = sent_tokenize(self.all_text)
        #Iterate through the list of sentences
        for sentence in sentences:
            #Tokenize the words within the sentence
            tokens = word_tokenize(sentence)
            #Remove Puncutation, lower case word, and lemmatize the word
            words = [self.lemma.lemmatize(word.lower()) for word in tokens if word.isalpha()]
            #Find the trigrams within the sentence
            trigrams = ngrams(words,3)
            #Add the tri-grams to the count dictionary
            self.tri_counts += Counter(trigrams)   
        #Hold the total count of all tri-grams
        self.total = sum(self.tri_counts.values())   
 
    #Find the probability of a sentence
    def sentence_prob(self,sentence):
        sent_tokens = self.sentence_tokens(sentence)
        #Start with prob of 1.0 since all probs are multiplied since
        #this is assuming "and":  P(tri-grams) = P(tri-gram1) * P(tri-gram2) * ...
        prob = 1.0
        #Iterate through all trigrams
        for tri_gram in ngrams(sent_tokens,3):
            prob *= (float(self.tri_counts.get(tri_gram,0))/self.total)
        return prob

    #Gather the tokens of a sentence
    def sentence_tokens(self,sentence):
        sent_tokens = [self.lemma.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.isalpha()]
        return sent_tokens

    #Find the probability of text
    def text_prob(self,text):
        sentences = sent_tokenize(text)
        prob = 1.0
        for sentence in sentences:
            prob *= self.sentence_prob(sentence)
        return prob

    #Find the entropy of text
    def entropy(self,text):
        probs = []
        for sentence in sent_tokenize(text):
            sent_num_words = len([self.lemma.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.isalpha()])
            sentence_prob = self.sentence_prob(sentence)
            if float(sent_num_words) == 0 or sentence_prob == 0:
                probs.append(1)
            else:
                probs.append(-1*np.log(sentence_prob) / float(sent_num_words))
        if len(probs) == 0:
            return 1
        avg_entropy = sum(probs) / float(len(probs))
        return avg_entropy

    #Find the prototypicality of text
    def prototypicality(self, text):
        #text = text.decode(self.encoding)
        entropy = self.entropy(text)
        prototypicality = -1*entropy
        return prototypicality

if __name__ == "__main__":
    #Expect first argument to sys.argv is a csv filename
    #This csv file is expected to have headers and a column
    #header named "text"
    text_file = open(sys.argv[1],"rb")

    all_text = ""
    
    #Iterate through csv file to gather all text to build the language model
    reader = csv.DictReader(text_file)
    for idx,row in enumerate(reader):
        text = unicode(row["text"], "latin-1").replace(u'\xa0',u'')
        all_text += text + " "
            
    #Initialize language model
    stat_lang_model = SLM(all_text,encoding="latin-1")

    #Reopen text file to find metrics for each piece of text
    print("Iterating through text")
    with open(sys.argv[1],"rb") as text_file:
        reader = csv.reader(text_file)
        #Set new file writer
        new_file_name = sys.argv[1].split(".csv")[0] + "_StatLang_Results.csv"
        new_file = open(new_file_name,"wb")
        writer = csv.writer(new_file)
        
        for idx,row in enumerate(reader):
            if idx < 1:
                headers = row
                #Append new metric columns
                headers.append("Entropy")
                headers.append("Prototypicality")
                writer.writerow(headers)
            else:
                text = unicode(row[headers.index("text")], "latin-1").replace(u'\xa0',u'')
                row.append(stat_lang_model.entropy(text))
                row.append(stat_lang_model.prototypicality(text))
                writer.writerow(row)
                
        new_file.close()



