import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

#Assign the files to variables
file = 'medical_raw_data_original.csv'
filecleanEnv = 'medical_raw_data_clean.csv'
fileExported = 'medical_raw_data_cleaned_export.csv'
#Read the file into the Dataframes df and df3
df = pd.read_csv(file, sep=',',comment='#',na_values=['NA'])
df3 = pd.read_csv(filecleanEnv, sep=',',comment='#',na_values=['NA'])
df4 = pd.read_csv(fileExported, sep=',',comment='#',na_values=['NA'])

#Functions for each part of the project
def D206PartTwoDetectionCode():
    #Check for null values by field
    print(df.info())
    #Check for duplicates
    print(df.duplicated())
    print("Number of Duplicate Rows: " + str(df.duplicated().sum()))
    print("Number of Unique Rows: " + str((~df.duplicated()).sum()))
    #Histograms to check for Anomalies
    Column_Name = input("Enter a Column Name to Examine with Histogram\n")
    #Use histogram to look at certain Columns (such as 
    #Children, Age, Income, Overweight, Anixety, Initial_days)
    plt.hist(df[(str(Column_Name))])
    plt.title(str(Column_Name))
    #Check all columns in histogram form
    df.hist()
    #Chart to See Anomalies in visual form when it comes to null NA values
    msno.matrix(df,labels=True)

def D206PartThreeTreatmentCode():
    #Use the global command to change the global df3 instead of just in the function
    global df3
    #Input what column the user wants to change
    ColumnToChange = input("Enter the name of the column you'd like to fix NAs on\n")
    #Input what the user wants to do to treat the NA's
    TreatNAs = input("How do you want to treat the NA's?(Mean,Median,Mode)\n")
    #Treat each data point that has NAs 
    if TreatNAs == 'Mean':
        df3[ColumnToChange].fillna(df3[ColumnToChange].median(),inplace=True)
    if TreatNAs == 'Median':
        df3[ColumnToChange].fillna(df3[ColumnToChange].median(),inplace=True)
    if TreatNAs == 'Mode':
        df3[ColumnToChange]=df3[ColumnToChange].fillna(df3[ColumnToChange].mode()[0])
    plt.hist(df3[(str(ColumnToChange))])
    plt.title(str(ColumnToChange))

def D206PartThreeProvideCode():
    global df3
    df3['Children'].fillna(df3['Children'].median(),inplace=True)
    df3['Age'].fillna(df3['Age'].mean(),inplace=True)
    df3['Income'].fillna(df3['Income'].median(),inplace=True)
    df3['Children'].fillna(df3['Children'].median(),inplace=True)
    df3['Initial_days'] = df3 ['Initial_days'].fillna(df3['Initial_days'].mode()[0])
    df3['Soft_drink']=df3['Soft_drink'].fillna(df3['Soft_drink'].mode()[0])
    df3['Overweight']=df3['Overweight'].fillna(df3['Overweight'].mode()[0])
    df3['Anxiety']=df3['Anxiety'].fillna(df3['Anxiety'].mode()[0])
    df3.to_csv('medical_raw_data_cleaned_export.csv')

def D206PartThreeCheckVariablesAfterClean():
    #Assign the file to variable
    fileclean = 'medical_raw_data_cleaned_export.csv'
    #Read the file into the Dataframes df and df3
    df2 = pd.read_csv(fileclean, sep=',',comment='#',na_values=['NA']) 
    #Check for null values by field
    print(df2.info())
    #Check for duplicates
    print(df2.duplicated())
    print("Number of Duplicate Rows: " + str(df2.duplicated().sum()))
    print("Number of Unique Rows: " + str((~df2.duplicated()).sum()))
    #Histograms to check for Anomalies
    Column_Name = input("Enter a Column Name to Examine with Histogram\n")
    #Use histogram to look at certain Columns (such as 
    #Children, Age, Income, Overweight, Anixety, Initial_days)
    plt.hist(df2[(str(Column_Name))])
    plt.title(str(Column_Name))
    #Chart to See Anomalies in visual form when it comes to null NA values
    msno.matrix(df2,labels=True)

def D206Part3EPCA():
    test_pca = pd.read_csv('medical_raw_data_cleaned_export.csv', index_col= 0)
    test_pca = test_pca[['Population', 'Children', 'Age', 'Income', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten','VitD_supp', 'Overweight', 'Anxiety', 'Initial_days', 'TotalCharge', 'Additional_charges']]
    test_pca_normalized = (test_pca-test_pca.mean())/test_pca.std()
    pca = PCA(n_components=test_pca.shape[1])
    pca.fit(test_pca_normalized)
    test_pca = pd.DataFrame(pca.transform(test_pca_normalized), 
    columns=['Population', 'Children', 'Age', 'Income', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten','VitD_supp', 'Overweight', 'Anxiety', 'Initial_days', 'TotalCharge', 'Additional_charges'])
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('explained variance')
    plt.show()
    cov_matrix = np.dot(test_pca_normalized.T,test_pca_normalized)/test_pca.shape[0]
    eigenvalues = [np.dot(eigenvector.T,np.dot(cov_matrix, eigenvector)) for 
    eigenvector in pca.components_]
    plt.plot(eigenvalues)
    plt.xlabel('number of components')
    plt.ylabel('eigenvalue')
    plt.show()
    loadings = pd.DataFrame(pca.components_.T,
    columns=['Population', 'Children', 'Age', 'Income', 'VitD_levels', 'Doc_visits', 'Full_meals_eaten','VitD_supp', 'Overweight', 'Anxiety', 'Initial_days', 'TotalCharge', 'Additional_charges'],
    index=test_pca.columns)
    #make sure all columns and rows are displayed in pandas
    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_rows',None)
    plt.axhline(y=1,color='r',linestyle='-')
    plt.show()
    loadings
    print(loadings)
    
    
#Use below as needed to call a function, or call them all
D206PartTwoDetectionCode()
#D206PartThreeTreatmentCode()
#D206PartThreeProvideCode()
D206PartThreeCheckVariablesAfterClean()
#D206Part3EPCA()

