'''
Program to find the decriptive statistics for hostels
'''

import pandas as pd 
import re
import matplotlib.pyplot as plt
import seaborn as sns
import data_process as dp
import ngram
import pydotplus


# Part 1: Distribution of Traveller types in hostels versus hotels over time
def extract_year(string):
	'''
	Extract year given date string
	'''
	return re.findall(r'(\d{4})', string)[0]

def dist(csv_file):
	'''
	Finds the distribution of each traveler type given the data. 
	'''
	data = pd.read_csv(csv_file)
	data['Year'] = data["Date"].apply(extract_year)
	data = data[data["Traveller Type"] != 'Family']

	df_new = data.groupby(['Year', 'Traveller Type']).count()['Hotel Name'].reset_index()

	sns.axes_style("white")
	ax = sns.catplot(x="Year", y="Hotel Name", hue="Traveller Type", \
		kind="bar", palette="RdPu", data=df_new)
	ax.fig.suptitle('Distribution of Traveller Types')
	ax.set(xlabel='Year', ylabel='Total Count')

	plt.savefig('type_dist.png')


def get_toplist(data, traveltype):
	data = data[data['Traveller Type'] == 'Solo']
	processed_data = ngram.data_process(data)
	top_list = ngram.overall_top(processed_data)

	return top_list

# Part 2: Visualize the bi-grams

def trigram_plot(csv_file):
	'''
	Plots charts for each traveler type

	To set the 1,2 or 3 gram or stem/not stem - modify ngram.py most_freq
	'''

	#Create New Dataframe with top list of words identified for each traveller type.
	data = pd.read_csv(csv_file)

	data_solo = data[data['Traveller Type'] == 'Solo']
	data_friends = data[data['Traveller Type'] == 'Friends']
	data_family = data[data['Traveller Type'] == 'Family']
	data_business = data[data['Traveller Type'] == 'Business']
	data_couple = data[data['Traveller Type'] == 'Couple']

	solo_data = ngram.data_process(data_solo)
	top_solo = ngram.overall_top(solo_data)

	friends_data = ngram.data_process(data_friends)
	top_friends = ngram.overall_top(friends_data)

	family_data = ngram.data_process(data_family)
	top_family = ngram.overall_top(family_data)

	business_data = ngram.data_process(data_business)
	top_business = ngram.overall_top(business_data)

	couple_data = ngram.data_process(data_couple)
	top_couple = ngram.overall_top(couple_data)

	top_list = top_solo + top_couple + top_family + top_business + top_friends

	top_df = pd.DataFrame(top_list)
	top_df.columns = ['n-grams', 'Count']
	
	top_df['Type'] = 'None'
	top_df.loc[0:10, 'Type'] = 'Solo'
	top_df.loc[11:20, 'Type'] = 'Couple'
	top_df.loc[21:30, 'Type'] = 'Family'
	top_df.loc[31:40, 'Type'] = 'Business'
	top_df.loc[41:50, 'Type'] = 'Friends'
	

	#Use the data frame to create plot
	#Adjust title, and ngram file based on the output desired.  

	f, ax = plt.subplots(ncols=2, nrows=2, figsize=(50, 50))

	left   =  0.23  # the left side of the subplots of the figure
	right  =  0.9    # the right side of the subplots of the figure
	bottom =  0.1    # the bottom of the subplots of the figure
	top    =  0.9    # the top of the subplots of the figure
	wspace =  .9    # the amount of width reserved for blank space between subplots
	hspace =  0.4    # the amount of height reserved for white space between subplots

	# This function actually adjusts the sub plots using the above paramters
	plt.subplots_adjust(
	    left    =  left, 
	    bottom  =  bottom, 
	    right   =  right, 
	    top     =  top, 
	    wspace  =  wspace, 
	    hspace  =  hspace
	)

	plt.suptitle("Top Lancaster Stemmed Trigrams For Each Traveller Type", fontsize=25)

	ax[0][0].set_title("Solo")
	ax[0][1].set_title("Friends")
	ax[1][0].set_title("Business")
	ax[1][1].set_title("Couple")

	sns.set(font_scale=1.4)
	sns.set_style("whitegrid")
	sns.barplot(x="Count", y="n-grams", data=top_df.loc[0:10, :], ax= ax[0][0], color="palevioletred", ci=None)
	sns.barplot(x="Count", y="n-grams", data=top_df.loc[41:50, :], ax= ax[0][1], color="palevioletred", ci=None)
	sns.barplot(x="Count", y="n-grams", data=top_df.loc[31:40, :], ax= ax[1][0], color="palevioletred", ci=None)
	sns.barplot(x="Count", y="n-grams", data=top_df.loc[11:20, :], ax= ax[1][1], color="palevioletred", ci=None)

	#Attempt to Sort Results

	# sns.barplot(x="Count", y="n-grams", data=top_df[top_df['Type'] == 'Solo'].sort_values(['Count']).reset_index(drop=True), ax= ax[0][0], color="darkmagenta", ci=None)
	# sns.barplot(x="Count", y="n-grams", data=top_df[top_df['Type'] == 'Friends'].sort_values(['Count']).reset_index(drop=True), ax= ax[0][1], color="darkmagenta", ci=None)
	# sns.barplot(x="Count", y="n-grams", data=top_df[top_df['Type'] == 'Business'].sort_values(['Count']).reset_index(drop=True), ax= ax[1][0], color="darkmagenta", ci=None)
	# sns.barplot(x="Count", y="n-grams", data=top_df[top_df['Type'] == 'Couple'].sort_values(['Count']).reset_index(drop=True), ax= ax[1][1], color="darkmagenta", ci=None)

	ax[0][0].set_xlabel("")
	ax[0][1].set_xlabel("")
	ax[1][0].set_xlabel("Count")
	ax[1][1].set_xlabel("Count")

	ax[0][0].set_ylabel("Tri-Grams")
	ax[0][1].set_ylabel("")
	ax[1][0].set_ylabel("Tri-Grams")
	ax[1][1].set_ylabel("")

	plt.show()
	plt.savefig('ngram.png')

# Part 3: td-idf - NOT DONE

# Part 4: LDA - NOT DONE


