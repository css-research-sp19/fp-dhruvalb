'''
DHRUVAL BHATT

#References: 
#https://gist.github.com/benhoyt/dfafeab26d7c02a52ed17b6229f0cb52
#https://chrisalbon.com/machine_learning/preprocessing_text/remove_stop_words/
https://stackoverflow.com/questions/38805341/plot-most-frequent-words-in-python
'''
import pandas as pd
import re
import collections
import sys
import time
from nltk.corpus import stopwords
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

nltk.download('stopwords')

#Basic Workflow: 
'''
Process data to get a df with tokenized words and ngrams for each review entry
find the top 10 words by comparing all ngrams from all reviesws to overall_top 10 

'''

#Aux functions for creating n-grams
def most_freq(dic, length=3):
	'''
	change length parameter to get ngrams stored for that
	length.
	'''
	cnt = dic[length]
	return cnt


def overall_top(data):
	'''
	Function to Find Top N Grams
	'''
	all_val = collections.Counter()
	for i in range(0, len(data)):
		all_val = all_val + data['counters'].values[i]
	
	return all_val.most_common(10)

def count_ngrams(words, min_length=1, max_length=3):
    '''
    Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    '''
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1

    # Loop through all lines and words and add n-grams to dict
    for word in words:
        #for word in tokenize(line):
        queue.append(word)
        if len(queue) >= max_length:
            add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams

def remove_numbers(string):
	'''
	Removes any numbers
	'''
	return re.sub(r'\d+', '', string)

def apply_stem(words):
    #porter = PorterStemmer()
    porter = LancasterStemmer()
    ret_words = []

    for word in words:
        ret_words.append(porter.stem(word))

    return ret_words

def data_process(csv_file):
	#data = pd.read_csv(csv_file)
	data = csv_file
	data['Text'] = data['Text'].str.lower()
	data['Text'] = data['Text'].str.replace('[^\w\s]','')
	data['Text'] = data['Text'].apply(remove_numbers)
	

	# Tokenize
	def tokenize(string):
		return re.findall(r'\w+', string)
	data['Tokens'] = data["Text"].apply(tokenize)
	
	# Remove Stop Words
	def remove_stop(words):
		stop_words = stopwords.words('english')
		domain_words = set(['san', 'francisco', 'green', 'tortoise', 'hostel', 'orange', 'village'])
		domain_words.update(['ive', 'would', 'hi', 'center', 'city', 'really', 'hostels'])
		rem_words = []
		for word in words:
			if word not in stop_words and word not in domain_words:
				rem_words.append(word)
		return rem_words

	data['Tokens'] = data['Tokens'].apply(remove_stop)
	data['Tokens'] = data['Tokens'].apply(apply_stem)
	data['ngrams'] = data['Tokens'].apply(count_ngrams)
	data['counters'] = data['ngrams'].apply(most_freq)
	
	return data



def print_most_frequent(ngrams, num=10):
	"""Print num most common n-grams of each length in n-grams dict."""
	for n in sorted(ngrams):
	    print('----- {} most common {}-grams -----'.format(num, n))
	    for gram, count in ngrams[n].most_common(num):
	        print('{0}: {1}'.format(' '.join(gram), count))
	    print('')


def plot_freq(top_list):
    '''
    Create plots for the top words, given a list of words (created by most_freq)
    '''
	plt.clf()
	names, values = zip(*top_list)
	ind = np.arange(len(top_list))  # the x locations for the groups
	width = 0.7       # the width of the bars

	fig, ax = plt.subplots()
	ax.barh(ind, values, width, color="salmon")
	ax.set_xlabel('Count')
	ax.set_yticks(ind+width/2.)
	ax.set_yticklabels(names)
	ax.set_title('The Most Common Trigrams for Reviews: Business')

	plt.show()


# LDA - NOT USED IN ANALYSIS

