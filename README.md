# Test for EA Sport


## Premise

The developed of this test was done with colab in a Google Drive platform if you want acces to the drive and run these notebooks ask at ediboc@gmail.com (only test evaluators)

The transformation of data was done with pandas library  and the final tables were import to MySQL . This can be done manually or with the python library mysql.

Here we make a resume for the steps done, all the code was developed in jupyter notebooks and there you will find comments for further details

## Objective 1: Localizate the game by language

#### Notebook: 1-1-joinCSV.ipynb

In this notebook we joined the 3 cvs files into one table, after cleaning the data with: elimination of duplicate row, fix numeric variables and wrong values.

resulting files: 
*CompleteCSVDataset.csv:* is the result of join PlayerPersonalData.csv, PlayerAttributeData.csv and PlayerPersonalData.
*ColumnDataIndex.csv:* is a table or dataframe, that contains all the column names from CompleteCSVDataset.csv in the first column and the original table where the column comes from is in the second column.

#### Notebook: 1-2-join_JSON.ipynb

In this notebook the 3 json files were joined into one table, the json files were transform into a list of document and the function json_normalize was used to transform it in dataframes

resulting file: 
*CompleteJSONDataset.csv:* is the result of join languages.json, countries.json and continents.json in one data frame.

#### Notebook: 1-3ExploratoryAnalisys.ipynb
This notebook join the files CompleteCSVDataset.csv and CompleteJSONDataset.csv and make analysis of metrics by language

resulting file: 
*CompleteDataLanguage.csv:* is the join of players files (CSV) and language (JSON).


#### EA-test-database.sql: sqlite database
Here we export to tables CompCsvClusters.csv as players (primary key ID) and CompleteJSONDataset (primary key ID_ctlg) as country_languages.
The data observation can be store in a row, the tables have primary key and do not have sub documents, then we can choose a structured database.
 
## Objective 2:Build a Data Visualization of the data FIFA game by country and languages. Clustering to help us for take the best decisions for Localization.

It is important to highlight that there is a big difference in the number of observations between countries or languages, so it is decided to clustering with the best 20 players from each country or language.

#### Notebook: 2-1Cluster_countries.ipynb
Clustering by country and top 20 players
We get 5 clusters where cluster 1 is the countries that produces the best players and whose performance is similar.
And the cluster 2 of countries with similar performance and metrics like Overall and Potential slightly smaller than those in group 1.
This is the same analysis for groups 3, 4 and 5.

resulting file: 
*CompCsvCLusters.csv:* is the CompleteCSVDataset.csv with the column Cluster_countries added, for players not in the top 20 or others Cluster_countries is 0. 

#### Notebook: 2-2Cluster_languages.ipynb
Clustering by language and top 20 players

We get 7 clusters where cluster 1 are the languages that produces the best players and whose performance is similar.

resulting file: 
*languageCLusters.csv*: is dataframe of languagues average metrics and Cluster_language

#### Qlik dashboard: EA-test_Countries_Clusters.qvf
To display metrics of players, teams and countries based on country clusters.

#### Qlik dashboard: EA-test_Language_Clusters.qvf
To display metrics by lannguage clusters.

Note: there is a file 'dashboard print screen.doc'

## Objective 3: Sentiment Analysis of FIFA20

In order to get the sentiment analysis:
1) First we have to classify the comments as positive or negative.
2) Get comments from a social network
3) Classify comments and see results


### 3.1 Generate a classifier: 
In the folder NLP_movie we will create a classifier based on nltk movie_reviews data
On notebook **NLP_preprocess.ipynb** is the preprocess of the documents and **NLP_Models.ipynb** is the model generation (LogisticRegression, XGBClassifier, SVC from Sklearn and a Neural Network), where we chose the neural network model because we obtained the highest accuracy.

resulting file: 
*NN_model.h5*:Neural network model used to classify comments

### 3.2 Get comments from Twitter
Notebook **Get_tweets.ipynb** where we will get tweets and their location related to FIFA20, saved in the file tweetsDB.csv

resulting file: 
*DFtweets.csv:*  dataframe of tweets with info of user, location, date of creation and text

### 3.3  Classify comments and see results
Notebook **3-1tweets_comments_classifier.ipynb**, where we preprocess tweet texts, it is classified as positive or negative comments and a table of results is presented by location 

### 3.4 additiona: get comments from youtube videos and classify them
This comments have no information about location

Notebook **Get_youtube_comments.ipynb** where we will get comments in videos related to FIFA20, saved in the file ytComments_translated.csv.

Notebook **3-2youtube_comments_classifier.ipynb** where we preprocess videos comments, it is classified as positive or negative comments and a table of results is presented by video.

#### Issues and Improvement opportunities: 
It can be found on a word document with the same name
