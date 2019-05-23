'''
'''
import pandas as pd
import re
import collections
import sys
import time
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Obtain your text sources
# Extract documents and move into a corpus
# Transformation
# Extract features
# Perform analysis

#Load data

def data_prep(csv_file):
	data = pd.read_csv(csv_file)
	data['Text'] = data['Text'].str.lower()
	data['Text'] = data['Text'].str.replace('[^\w\s]','')

	# Tokenize
	def tokenize(string):
		return re.findall(r'\w+', string)

	#Remove Stop Words
	data['Text'] = data["Text"].apply(tokenize)
	#print(data['Text'][0])

	return data


def count_ngrams(lines, min_length=2, max_length=4):
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
    for line in lines:
        for word in tokenize(line):
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams

def print_most_frequent(ngrams, num=10):
	"""Print num most common n-grams of each length in n-grams dict."""
	for n in sorted(ngrams):
	    print('----- {} most common {}-grams -----'.format(num, n))
	    for gram, count in ngrams[n].most_common(num):
	        print('{0}: {1}'.format(' '.join(gram), count))
	    print('')

ngrams = count_ngrams(data['Text'][0])

data['ngrams'] = data['Text'].apply(count_ngrams)

print(data['ngrams'].head())

#print_most_frequent(ngrams)


# Convert to lower case
# Remove punctuation
# Remove numbers
# Remove stopwords
# Remove domain-specific stopwords
# Name of hotel, booked 
# Stemming


#References: 
#https://gist.github.com/benhoyt/dfafeab26d7c02a52ed17b6229f0cb52
#https://chrisalbon.com/machine_learning/preprocessing_text/remove_stop_words/
