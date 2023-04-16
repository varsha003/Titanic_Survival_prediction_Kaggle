The challenge : Predict Survival on the Titanic
The first and best challenge to start into ML project is Titanic survival. Every Data scientist and ML engineer had started their journey with this project.

The challenge is very simple - Need to predict which passenger survived during titanic shipwreck.

Check out : Kaggle titanic

Dataset that we are going to use : you can download it from Titanic data and create notebook in kaggle platform or can download from GITHUB

1.train.csv

2.test.csv

3.gender_submission.csv

Let’s see what’s in the dataset

train.csv - contains details of the passenger and whether they survived or not know as “ground truth”

test.csv- contains details of the passenger and we need to predict “ground truth” outcomes.

gender_submission.csv - sample submission of outcomes and format of submission

LET’S DIVE
You can create notebook in kaggle or use your comfortable IDE.

Let’s start with EDA :

Import package and read the data set .

## Import the package import pandas as pd import numpy as np

## read the dataset train = pd.read_csv('input/train.csv') test= pd.read_csv('input/test.csv')

Check shape and info

# shape of the dataset

print ('train shape', train.shape)

print('test shape',test.shape)

train shape (891, 12) test shape (418, 11)

train.info()

test.info()

Data Dictionary
Survived: 0 = No, 1 = Yes

pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd

sibsp: # of siblings / spouses aboard the Titanic

parch: # of parents / children aboard the Titanic

ticket: Ticket number

cabin: Cabin number

embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

Total rows and columns

We can see that there are 891 rows and 12 columns in our training dataset.

Missing value
train.isnull().sum()

test.isnull().sum()

## create a column survival in test data

test['survival']=""

test.head(10)

Data Visualization using Matplotlib and Seaborn packages.
import matplotlib.pyplot as plt # Plot the graphes

%matplotlib inline

import seaborn as sns

sns.set() # setting seaborn default for plots

def bar_chart(feature):

survived = train[train['Survived']==1][feature].value_counts()

dead = train[train['Survived']==0][feature].value_counts()

df = pd.DataFrame(['survived','dead'])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(10,5))

checking survival with all features

bar_chart('Sex')

print("Survived :\n",train[train['Survived']==1]['Sex'].value_counts())

print("Dead:\n",train[train['Survived']==0]['Sex'].value_counts())


The Chart confirms Women more likely survivied than Men.

Checking with Pclass

bar_chart('Pclass')

print("Survived :\n",train[train['Survived']==1]['Pclass'].value_counts())

print("Dead:\n",train[train['Survived']==0]['Pclass'].value_counts())



The Chart confirms 1st class more likely survivied than other classes. The Chart confirms 3rd class more likely dead than other classes

Same way check with all the categorical feature

sibsp: # of siblings / spouses aboard the Titanic

parch: # of parents / children aboard the Titanic

embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

For sibsp: The Chart confirms a person aboard with more than 2 siblings or spouse more likely survived. The Chart confirms a person aboard without siblings or spouse more likely dead.

For parch: The Chart confirms a person aboard with more than 2 parents or children more likely survived.
The Chart confirms a person aboard alone more likely dead

For embarked :The Chart confirms a person aboard from C slightly more likely survived. The Chart confirms a person aboard from Q more likely dead. The Chart confirms a person aboard from S more likely dead.

Feature engineering
Feature engineering is the process of using domain knowledge of the data to create features (feature vectors) that make machine learning algorithms work.

feature vector is an n-dimensional vector of numerical features that represent some object. Many algorithms in machine learning require a numerical representation of objects, since such representations facilitate processing and statistical analysis.

combine the dataset

train_test_data= [train,test]

for dataset in train_test_data: ## checking the titles

dataset['Title']=dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()

test['Title'].value_counts()

Title Map
Mr : 0
Miss : 1
Mrs: 2
Others: 3

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3, "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

dataset['Title'] = dataset["Title"].map(title_mapping)

delete unnecessary feature from dataset

train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)

Converting Sex categorical column to numerical

Before : Male & female

After : 0 & 1

Male =0, Female = 1

CODE:

sex_mapping = {"male": 0, "female": 1}

for dataset in train_test_data:

dataset['Sex'] = dataset['Sex'].map(sex_mapping)

Filling out missing values in Age

train["Age"].fillna(train.groupby("Title")"Age"].transform("median"),inplace=True)

test["Age"].fillna(test.groupby("Title")"Age"].transform("median"),inplace=True)

Age insights

facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.show()


Those who were 20 to 30 years old were more dead and more survived.

Binning

Binning/Converting Numerical Age to Categorical Variable

feature vector map:

child: 0 young: 1 adult: 2 mid-age: 3 senior: 4

for dataset in train_test_data:

dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1

dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2

dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3

dataset.loc[dataset['Age'] > 62, 'Age'] = 4


Same way follow for Embarked column convert into numerical groupby Pclass

more than 50 % of 1st class are from S embark. more than 50 % of 2st class are from S embark. more than 50 % of 3st class are from S embark.

fill out missing embark with S embark

for dataset in train_test_data:

dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {'S':0,'C':1,'Q':2}

for dataset in train_test_data:

dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# fill missing Fare with median fare for each Pclass

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

train.head(50)

for dataset in train_test_data:

dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0

dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1

dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2

dataset.loc[dataset['Fare'] >= 100, 'Fare'] = 3

Combine Sibsp and Parch column as family size and follow the sample steps

MODELLING
We are using all the classification model as the outcome of the survival is either 0 or 1.

# Importing Classifier Modules

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import numpy as np

Cross Validation(k-fold)
Checking out the accuracy of all the classification algorithm. The model which gives highest accuracy is taken .

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

On basis of model score SVM algorithm gives maximum accuracy and using SVM algorithm to predict .

test_data['Survived'] = prediction

submission = pd.DataFrame(test['PassengerId'],test_data['Survived'])

submission.to_csv("Submission.csv")

Submit the outcome in excel format in kaggle

For more information check out GITHUB
