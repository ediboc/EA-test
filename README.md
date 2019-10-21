# Test for EA Sport


## Premise

The developed of this test was done with colab in a Google Drive platform if you want acces to the drive and run these notebooks ask at ediboc@gmail.com (only test evaluators)

The transformation of data was done with pandas library  and the final tables were import to sqlite . This can be done manually or with the library sqlite3.

Here we make a resume for the steps done, all the code was developed in jupyter notebooks and there you will find comments for further details

## Objective 1

#### Notebook: 1-1-joinCSV.ipynb

In this notebook we joined the 3 cvs files into one table, after cleaning the data with: elimination of duplicate row, fix numeric variables and wrong values.

CompleteCSVDataset.csv: is the result of join PlayerPersonalData.csv, PlayerAttributeData.csv and PlayerPersonalData.
ColumnDataIndex.csv: is a table or dataframe, that contains all the column names from CompleteCSVDataset.csv in the first column and the original table where the column comes from is in the second column.

#### Notebook: 1-2-join_JSON.ipynb

In this notebook the 3 json files were joined into one table, the json files were transform into a list of document and the function json_normalize was used to transform it in dataframes

CompleteJSONDataset.csv: is the result of join languages.json, countries.json and continents.json in one data frame.

#### Notebook: 1-3ExploratoryAnalisys.ipynb
This notebook join the files CompleteCSVDataset.csv and CompleteJSONDataset.csv and make analysis of metrics by language
