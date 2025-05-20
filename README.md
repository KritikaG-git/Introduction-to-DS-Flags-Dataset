# COS60008 Introduction to Data Science
## Assignment 2, 2023, Semester 1
## Student Details:
* Name: Kritika.
* Student Id: 104361002
* Email: mailto:104361002@student.swin.edu.au
* Submission Date: 29/05/23
* TuteLab Class: Thursday 3:30 - 5:30pm.

## Task 1: Problem Formulation, Data Acquisition and Preparation

## Data Source url: https://archive.ics.uci.edu/ml/datasets/Flags

### This assignment is done on JupyterLab Notebook which can be found in Anaconda Navigator.


```python
import pandas as pd
#the column names are found in the description of the datset 
column_names = [
    'name', 'landmass', 'zone', 'area', 'population', 'language', 'religion', 'bars',
    'stripes', 'colours', 'red', 'green', 'blue', 'gold', 'white', 'black', 'orange',
    'mainhue', 'circles', 'crosses', 'saltires', 'quarters', 'sunstars', 'crescent',
    'triangle', 'icon', 'animate', 'text', 'topleft', 'botright'
]


```


```python
#convert the dataset in to an acceptable format
df = pd.read_csv('flag.data', names=column_names, sep=',')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>landmass</th>
      <th>zone</th>
      <th>area</th>
      <th>population</th>
      <th>language</th>
      <th>religion</th>
      <th>bars</th>
      <th>stripes</th>
      <th>colours</th>
      <th>...</th>
      <th>saltires</th>
      <th>quarters</th>
      <th>sunstars</th>
      <th>crescent</th>
      <th>triangle</th>
      <th>icon</th>
      <th>animate</th>
      <th>text</th>
      <th>topleft</th>
      <th>botright</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>5</td>
      <td>1</td>
      <td>648</td>
      <td>16</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>black</td>
      <td>green</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>3</td>
      <td>1</td>
      <td>29</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>red</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>4</td>
      <td>1</td>
      <td>2388</td>
      <td>20</td>
      <td>8</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>green</td>
      <td>white</td>
    </tr>
    <tr>
      <th>3</th>
      <td>American-Samoa</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>blue</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>blue</td>
      <td>red</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>189</th>
      <td>Western-Samoa</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>blue</td>
      <td>red</td>
    </tr>
    <tr>
      <th>190</th>
      <td>Yugoslavia</td>
      <td>3</td>
      <td>1</td>
      <td>256</td>
      <td>22</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>blue</td>
      <td>red</td>
    </tr>
    <tr>
      <th>191</th>
      <td>Zaire</td>
      <td>4</td>
      <td>2</td>
      <td>905</td>
      <td>28</td>
      <td>10</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>green</td>
      <td>green</td>
    </tr>
    <tr>
      <th>192</th>
      <td>Zambia</td>
      <td>4</td>
      <td>2</td>
      <td>753</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>green</td>
      <td>brown</td>
    </tr>
    <tr>
      <th>193</th>
      <td>Zimbabwe</td>
      <td>4</td>
      <td>2</td>
      <td>391</td>
      <td>8</td>
      <td>10</td>
      <td>5</td>
      <td>0</td>
      <td>7</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>green</td>
      <td>green</td>
    </tr>
  </tbody>
</table>
<p>194 rows × 30 columns</p>
</div>



## Task 2: Data Exploration 

### Task 2.1


```python
#Descriptive statistics for 10 attributes. These are a mixture of numerical and categorical values.
import matplotlib.pyplot as plt
```


```python
landmassDescp = df['landmass'].describe()
print(landmassDescp)
zoneDescp = df['zone'].describe()
print(zoneDescp)
areaDescp = df['area'].describe()
print(areaDescp)
populationDescp = df['population'].describe()
print(populationDescp)
languageDescp = df['language'].value_counts()
print(languageDescp)
religionDescp = df['religion'].value_counts()
print(religionDescp)
barsDescp = df['bars'].value_counts()
print(barsDescp)
stripesDescp = df['stripes'].value_counts()
print(stripesDescp)
coloursDescp = df['colours'].value_counts()
print(coloursDescp)
mainhueDescp = df['mainhue'].value_counts()
print(mainhueDescp)
```

    count    194.000000
    mean       3.572165
    std        1.553018
    min        1.000000
    25%        3.000000
    50%        4.000000
    75%        5.000000
    max        6.000000
    Name: landmass, dtype: float64
    count    194.000000
    mean       2.211340
    std        1.308274
    min        1.000000
    25%        1.000000
    50%        2.000000
    75%        4.000000
    max        4.000000
    Name: zone, dtype: float64
    count      194.000000
    mean       700.046392
    std       2170.927932
    min          0.000000
    25%          9.000000
    50%        111.000000
    75%        471.250000
    max      22402.000000
    Name: area, dtype: float64
    count     194.000000
    mean       23.268041
    std        91.934085
    min         0.000000
    25%         0.000000
    50%         4.000000
    75%        14.000000
    max      1008.000000
    Name: population, dtype: float64
    10    46
    1     43
    6     30
    2     21
    8     19
    3     17
    4      6
    5      4
    7      4
    9      4
    Name: language, dtype: int64
    1    60
    0    40
    2    36
    5    27
    6    15
    3     8
    4     4
    7     4
    Name: religion, dtype: int64
    0    159
    3     21
    2      7
    1      6
    5      1
    Name: bars, dtype: int64
    0     110
    3      44
    2      16
    5      12
    1       4
    9       2
    11      1
    14      1
    4       1
    6       1
    13      1
    7       1
    Name: stripes, dtype: int64
    3    75
    4    44
    2    42
    5    15
    6     9
    7     7
    8     1
    1     1
    Name: colours, dtype: int64
    red       71
    blue      40
    green     31
    white     22
    gold      19
    black      5
    orange     4
    brown      2
    Name: mainhue, dtype: int64
    


```python
#Visualization

df['landmass'].value_counts().plot(kind='bar')
plt.xlabel('Landmass')
plt.ylabel('Frequency')
plt.title('Landmass distribution')

plt.savefig('Figure 1')
plt.show()
```


    
![png](output_11_0.png)
    



```python
df['zone'].value_counts().plot(kind='bar')
plt.xlabel('Zone')
plt.ylabel('Frequency')
plt.title('Zone distribution')

plt.savefig('Figure2')
plt.show()
```


    
![png](output_12_0.png)
    



```python
df['area'].plot(kind='hist', bins=10)
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.title('Area distribution')

plt.savefig('Figure 3')
plt.show()
```


    
![png](output_13_0.png)
    



```python
df['population'].plot(kind='hist', bins=10)
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.title('Population distribution')

plt.savefig('Figure 4')
plt.show()
```


    
![png](output_14_0.png)
    



```python
df['language'].value_counts().plot(kind='bar')
plt.xlabel('Language')
plt.ylabel('Frequency')
plt.title('Language distribution')

plt.savefig('Figure 5')
plt.show()
```


    
![png](output_15_0.png)
    



```python
df['religion'].value_counts().plot(kind='bar')
plt.xlabel('Religion')
plt.ylabel('Frequency')
plt.title('Religion distribution')

plt.savefig('Figure 6')
plt.show()
```


    
![png](output_16_0.png)
    



```python
df['bars'].value_counts().plot(kind='bar')
plt.xlabel('Number of Bars in the flag')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Bars in the flag')

plt.savefig('Figure 7')
plt.show()
```


    
![png](output_17_0.png)
    



```python
df['stripes'].value_counts().plot(kind='bar')
plt.xlabel('Number of Stripes in the flag')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Stripes in the flag')

plt.savefig('Figure 8')
plt.show()
```


    
![png](output_18_0.png)
    



```python
df['colours'].value_counts().plot(kind='bar')
plt.xlabel('Number of colours in the flag')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Colours in the flag')

plt.savefig('Figure 9')
plt.show()
```


    
![png](output_19_0.png)
    



```python
df['mainhue'].value_counts().plot(kind='bar')
plt.xlabel('Main Hue')
plt.ylabel('Frequency')
plt.title('Main Hue in the flag')

plt.savefig('Figure 10')
plt.show()
```


    
![png](output_20_0.png)
    


### Task2.2


```python
#relation between some columns and descriptive statistics.
selected_columns = ['area', 'population']
selected_df = df[selected_columns]

stats = selected_df.describe()
print(stats)

selected_df.plot.scatter(x='area', y='population')
plt.xlabel('Area')
plt.ylabel('Population')
plt.title('Relationship between Area and Population')

plt.savefig('Figure 10')
plt.show()
```

                   area   population
    count    194.000000   194.000000
    mean     700.046392    23.268041
    std     2170.927932    91.934085
    min        0.000000     0.000000
    25%        9.000000     0.000000
    50%      111.000000     4.000000
    75%      471.250000    14.000000
    max    22402.000000  1008.000000
    


    
![png](output_22_1.png)
    



```python
#next pair,
selected_columns = ['mainhue', 'colours']
selected_df = df[selected_columns]

stats = selected_df.describe()
print(stats)
```

              colours
    count  194.000000
    mean     3.463918
    std      1.300154
    min      1.000000
    25%      3.000000
    50%      3.000000
    75%      4.000000
    max      8.000000
    


```python
grouped_data = df.groupby('mainhue')['colours'].mean()
grouped_data.plot(kind='bar')

plt.xlabel('Main Hue')
plt.ylabel('Average Number of Colours')
plt.title('Relationship between Main Hue and Colours')

plt.savefig('Figure11.png')
plt.show()
```


    
![png](output_24_0.png)
    



```python
#another pair,
selected_columns = ['stripes', 'bars']
selected_df = df[selected_columns]

stats = selected_df.describe()
print(stats)
```

              stripes        bars
    count  194.000000  194.000000
    mean     1.551546    0.453608
    std      2.328005    1.038339
    min      0.000000    0.000000
    25%      0.000000    0.000000
    50%      0.000000    0.000000
    75%      3.000000    0.000000
    max     14.000000    5.000000
    


```python
plt.scatter(df['stripes'], df['bars'])
plt.xlabel('Number of Stripes')
plt.ylabel('Number of Bars')
plt.title('Relationship between Stripes and Bars')

plt.savefig('Figure 12.png')
plt.show()
```


    
![png](output_26_0.png)
    



    <Figure size 640x480 with 0 Axes>



```python
landmass_zone_stats = df.groupby(['landmass', 'zone']).size().reset_index(name='count')

landmass_zone_desc = landmass_zone_stats.describe()
print(landmass_zone_desc)
```

            landmass       zone      count
    count  13.000000  13.000000  13.000000
    mean    3.846154   2.538462  14.923077
    std     1.625123   1.265924  11.586354
    min     1.000000   1.000000   1.000000
    25%     3.000000   1.000000   7.000000
    50%     4.000000   3.000000  10.000000
    75%     5.000000   4.000000  19.000000
    max     6.000000   4.000000  39.000000
    


```python
landmass_zone_ct = pd.crosstab(df['landmass'], df['zone'])
# crosstab let you compute a table of frequencies or some summary statistics for two or more categorical attributes
landmass_zone_ct.plot(kind='bar', stacked=True)
plt.xlabel('Landmass')
plt.ylabel('Count')
plt.title('Relationship between Landmass and Zone')
plt.legend(title='Zone')

plt.savefig('Figure 13')
plt.show()
```


    
![png](output_28_0.png)
    



```python
language_landmass_counts = df.groupby('landmass')['language'].value_counts()
language_landmass_describe = df.groupby('landmass')['language'].describe()

print("Count of language by landmass:")
print(language_landmass_counts)
print("\nDescriptive statistics of language by landmass:")
print(language_landmass_describe)
```

    Count of language by landmass:
    landmass  language
    1         1           19
              2            9
              6            2
              3            1
    2         2           11
              1            3
              6            2
              3            1
    3         6           17
              4            6
              1            3
              5            3
              3            2
              9            2
              2            1
              10           1
    4         10          23
              3           12
              8            7
              1            6
              6            4
    5         10          16
              8           12
              6            4
              7            4
              9            2
              5            1
    6         1           12
              10           6
              3            1
              6            1
    Name: language, dtype: int64
    
    Descriptive statistics of language by landmass:
              count      mean       std  min  25%  50%   75%   max
    landmass                                                      
    1          31.0  1.677419  1.275071  1.0  1.0  1.0   2.0   6.0
    2          17.0  2.352941  1.455214  1.0  2.0  2.0   2.0   6.0
    3          35.0  5.142857  2.060024  1.0  4.0  6.0   6.0  10.0
    4          52.0  6.769231  3.495742  1.0  3.0  8.0  10.0  10.0
    5          39.0  8.487179  1.502135  5.0  8.0  8.0  10.0  10.0
    6          20.0  4.050000  4.160908  1.0  1.0  1.0  10.0  10.0
    


```python
language_landmass_counts = df.groupby('landmass')['language'].value_counts().unstack()

language_landmass_counts.plot.bar(stacked=True)
plt.xlabel('Landmass')
plt.ylabel('Count')
plt.title('Distribution of Languages by Landmass')
plt.legend(title='Language')

plt.savefig('Figure 15')
plt.show()
```


    
![png](output_30_0.png)
    



```python
#2.3 Proposing a question for explorative data analysis:
#What is the religion mostly followed according to the landmass,
#landmass: 1=N.America, 2=S.America, 3=Europe, 4=Africa, 4=Asia, 6=Oceania

religion_landmass_counts = df.groupby('landmass')['religion'].value_counts().unstack()
# Descriptive statistics
religion_landmass_counts_describe = religion_landmass_counts.describe()
```


```python
religion_landmass_counts.plot.bar(stacked=True)
plt.xlabel('Landmass')
plt.ylabel('Count')
plt.title('Distribution of Religious Affiliations by Landmass')
plt.legend(title='Religion')

print(religion_landmass_counts_describe)
plt.savefig('Figure 16')
plt.show()
```

    religion          0          1          2    3         4         5         6  \
    count      5.000000   5.000000   3.000000  1.0  3.000000   2.00000  3.000000   
    mean       8.000000  12.000000  12.000000  8.0  1.333333  13.50000  5.000000   
    std        5.787918   7.141428   9.848858  NaN  0.577350  17.67767  3.605551   
    min        2.000000   3.000000   1.000000  8.0  1.000000   1.00000  1.000000   
    25%        2.000000   7.000000   8.000000  8.0  1.000000   7.25000  3.500000   
    50%        9.000000  13.000000  15.000000  8.0  1.000000  13.50000  6.000000   
    75%       13.000000  16.000000  17.500000  8.0  1.500000  19.75000  7.000000   
    max       14.000000  21.000000  20.000000  8.0  2.000000  26.00000  8.000000   
    
    religion         7  
    count     2.000000  
    mean      2.000000  
    std       1.414214  
    min       1.000000  
    25%       1.500000  
    50%       2.000000  
    75%       2.500000  
    max       3.000000  
    


    
![png](output_32_1.png)
    


## Task 3: Data Modelling

### Task 3.1


```python
#importing the necessary sklearn libraries
# we are using KNN and decision trees as our models
from sklearn.model_selection import train_test_split
# select the features and target variable, I chose language as one.

X = df.drop('language', axis=1)
y = df['language']


```


```python
# Splitting data into training and test sets:
X_train_suite1, X_test_suite1, y_train_suite1, y_test_suite1 = train_test_split(X, y, test_size=0.5, random_state=42)
```


```python
X_train_suite2, X_test_suite2, y_train_suite2, y_test_suite2 = train_test_split(X, y, test_size=0.4, random_state=42)

```


```python
X_train_suite3, X_test_suite3, y_train_suite3, y_test_suite3 = train_test_split(X, y, test_size=0.2, random_state=42)

```

### Task 3.2


```python

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

```


```python

# Perform one-hot encoding on categorical variables, this process makes categorical values easy to handle during data processing.
X_encoded = pd.get_dummies(X)

# Split the data into training and test sets, Suite1
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.5, random_state=42)

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Confusion matrix
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix of Training Set:")
print(confusion_matrix_train)
print("Confusion Matrix of Test Set:")
print(confusion_matrix_test)

# Classification accuracy
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy of Training Set:", accuracy_train)
print("Accuracy of Test Set:", accuracy_test)

# Precision, Recall, and F1 score
report_train = classification_report(y_train, y_train_pred)
report_test = classification_report(y_test, y_test_pred)
print("Classification Report of Training Set:")
print(report_train)
print("Classification Report of Test Set:")
print(report_test)
```

    Confusion Matrix of Training Set:
    [[17  0  0  0  3  0  0  0  1]
     [ 3  5  1  0  2  0  0  0  0]
     [ 4  0  2  0  0  0  0  0  3]
     [ 0  1  1  0  1  0  0  0  0]
     [ 2  3  1  0  7  0  0  0  1]
     [ 1  0  0  0  1  0  0  0  0]
     [ 2  1  0  0  1  0  6  0  1]
     [ 0  0  0  0  0  0  1  0  2]
     [ 2  3  2  0  1  0  1  0 14]]
    Confusion Matrix of Test Set:
    [[17  0  0  0  0  4  0  1  0  0]
     [ 1  2  3  0  0  1  0  1  0  2]
     [ 1  1  1  0  0  2  0  2  0  1]
     [ 1  0  0  0  0  1  0  1  0  0]
     [ 0  2  0  0  0  2  0  0  0  0]
     [ 7  1  1  0  0  2  0  2  0  3]
     [ 1  0  0  0  0  1  0  0  0  0]
     [ 2  1  1  0  0  2  0  1  0  1]
     [ 0  0  0  0  0  0  0  0  0  1]
     [ 5  6  3  0  0  3  0  0  0  6]]
    Accuracy of Training Set: 0.5257731958762887
    Accuracy of Test Set: 0.29896907216494845
    Classification Report of Training Set:
                  precision    recall  f1-score   support
    
               1       0.55      0.81      0.65        21
               2       0.38      0.45      0.42        11
               3       0.29      0.22      0.25         9
               4       0.00      0.00      0.00         3
               6       0.44      0.50      0.47        14
               7       0.00      0.00      0.00         2
               8       0.75      0.55      0.63        11
               9       0.00      0.00      0.00         3
              10       0.64      0.61      0.62        23
    
        accuracy                           0.53        97
       macro avg       0.34      0.35      0.34        97
    weighted avg       0.49      0.53      0.50        97
    
    Classification Report of Test Set:
                  precision    recall  f1-score   support
    
               1       0.49      0.77      0.60        22
               2       0.15      0.20      0.17        10
               3       0.11      0.12      0.12         8
               4       0.00      0.00      0.00         3
               5       0.00      0.00      0.00         4
               6       0.11      0.12      0.12        16
               7       0.00      0.00      0.00         2
               8       0.12      0.12      0.12         8
               9       0.00      0.00      0.00         1
              10       0.43      0.26      0.32        23
    
        accuracy                           0.30        97
       macro avg       0.14      0.16      0.15        97
    weighted avg       0.27      0.30      0.27        97
    
    

    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
from sklearn.tree import DecisionTreeClassifier

#Decision Tree model
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

# Confusion matrix
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix of Training Set:")
print(confusion_matrix_train)
print("Confusion Matrix of Test Set:")
print(confusion_matrix_test)

# Classification accuracy
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy of Training Set:", accuracy_train)
print("Accuracy of Test Set:", accuracy_test)

# Precision, Recall, and F1 score
report_train = classification_report(y_train, y_train_pred)
report_test = classification_report(y_test, y_test_pred)
print("Classification Report of Training Set:")
print(report_train)
print("Classification Report of Test Set:")
print(report_test)
```

    Confusion Matrix of Training Set:
    [[21  0  0  0  0  0  0  0  0]
     [ 0 11  0  0  0  0  0  0  0]
     [ 0  0  9  0  0  0  0  0  0]
     [ 0  0  0  3  0  0  0  0  0]
     [ 0  0  0  0 14  0  0  0  0]
     [ 0  0  0  0  0  2  0  0  0]
     [ 0  0  0  0  0  0 11  0  0]
     [ 0  0  0  0  0  0  0  3  0]
     [ 0  0  0  0  0  0  0  0 23]]
    Confusion Matrix of Test Set:
    [[18  3  1  0  0  0  0  0  0  0]
     [ 0  9  1  0  0  0  0  0  0  0]
     [ 0  3  2  0  0  1  0  0  0  2]
     [ 0  0  1  1  0  1  0  0  0  0]
     [ 0  0  0  1  0  2  0  0  0  1]
     [ 4  1  4  3  0  0  0  0  1  3]
     [ 0  0  0  0  0  0  0  0  0  2]
     [ 0  0  0  0  0  0  0  6  0  2]
     [ 0  0  0  0  0  0  0  0  0  1]
     [ 6  0  2  0  0  2  0  0  0 13]]
    Accuracy of Training Set: 1.0
    Accuracy of Test Set: 0.5051546391752577
    Classification Report of Training Set:
                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00        21
               2       1.00      1.00      1.00        11
               3       1.00      1.00      1.00         9
               4       1.00      1.00      1.00         3
               6       1.00      1.00      1.00        14
               7       1.00      1.00      1.00         2
               8       1.00      1.00      1.00        11
               9       1.00      1.00      1.00         3
              10       1.00      1.00      1.00        23
    
        accuracy                           1.00        97
       macro avg       1.00      1.00      1.00        97
    weighted avg       1.00      1.00      1.00        97
    
    Classification Report of Test Set:
                  precision    recall  f1-score   support
    
               1       0.64      0.82      0.72        22
               2       0.56      0.90      0.69        10
               3       0.18      0.25      0.21         8
               4       0.20      0.33      0.25         3
               5       0.00      0.00      0.00         4
               6       0.00      0.00      0.00        16
               7       0.00      0.00      0.00         2
               8       1.00      0.75      0.86         8
               9       0.00      0.00      0.00         1
              10       0.54      0.57      0.55        23
    
        accuracy                           0.51        97
       macro avg       0.31      0.36      0.33        97
    weighted avg       0.44      0.51      0.46        97
    
    

    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
X_encoded = pd.get_dummies(X)

# Split the data into training and test sets,suite2
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.4, random_state=42)

#KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Confusion matrix
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix of Training Set:")
print(confusion_matrix_train)
print("Confusion Matrix of Test Set:")
print(confusion_matrix_test)

# Classification accuracy
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy of Training Set:", accuracy_train)
print("Accuracy of Test Set:", accuracy_test)

# Precision, Recall, and F1 score
report_train = classification_report(y_train, y_train_pred)
report_test = classification_report(y_test, y_test_pred)
print("Classification Report of Training Set:")
print(report_train)
print("Classification Report of Test Set:")
print(report_test)
```

    Confusion Matrix of Training Set:
    [[20  0  0  0  2  0  2  0  1]
     [ 3  5  2  0  1  0  1  0  2]
     [ 5  0  2  0  0  0  1  0  3]
     [ 0  2  1  0  1  0  0  0  0]
     [ 4  2  1  0  6  0  2  0  2]
     [ 2  0  0  0  0  0  0  0  0]
     [ 2  1  0  0  2  0  6  0  1]
     [ 0  0  0  0  0  0  1  0  2]
     [ 3  4  1  0  2  0  1  0 17]]
    Confusion Matrix of Test Set:
    [[14  0  0  0  0  2  0  1  0  1]
     [ 0  1  3  0  0  1  0  0  0  2]
     [ 0  1  1  0  0  2  0  0  0  2]
     [ 1  0  0  0  0  1  0  0  0  0]
     [ 1  2  0  0  0  1  0  0  0  0]
     [ 5  1  1  0  0  2  0  1  0  3]
     [ 1  0  0  0  0  1  0  0  0  0]
     [ 1  1  2  0  0  1  0  1  0  1]
     [ 0  0  0  0  0  0  0  0  0  1]
     [ 5  3  4  0  0  2  0  0  0  4]]
    Accuracy of Training Set: 0.4827586206896552
    Accuracy of Test Set: 0.2948717948717949
    Classification Report of Training Set:
                  precision    recall  f1-score   support
    
               1       0.51      0.80      0.62        25
               2       0.36      0.36      0.36        14
               3       0.29      0.18      0.22        11
               4       0.00      0.00      0.00         4
               6       0.43      0.35      0.39        17
               7       0.00      0.00      0.00         2
               8       0.43      0.50      0.46        12
               9       0.00      0.00      0.00         3
              10       0.61      0.61      0.61        28
    
        accuracy                           0.48       116
       macro avg       0.29      0.31      0.30       116
    weighted avg       0.43      0.48      0.45       116
    
    Classification Report of Test Set:
                  precision    recall  f1-score   support
    
               1       0.50      0.78      0.61        18
               2       0.11      0.14      0.12         7
               3       0.09      0.17      0.12         6
               4       0.00      0.00      0.00         2
               5       0.00      0.00      0.00         4
               6       0.15      0.15      0.15        13
               7       0.00      0.00      0.00         2
               8       0.33      0.14      0.20         7
               9       0.00      0.00      0.00         1
              10       0.29      0.22      0.25        18
    
        accuracy                           0.29        78
       macro avg       0.15      0.16      0.15        78
    weighted avg       0.25      0.29      0.26        78
    
    

    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

# Confusion matrix
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix of Training Set:")
print(confusion_matrix_train)
print("Confusion Matrix of Test Set:")
print(confusion_matrix_test)

# Classification accuracy
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy of Training Set:", accuracy_train)
print("Accuracy of Test Set:", accuracy_test)

# Precision, Recall, and F1 score
report_train = classification_report(y_train, y_train_pred)
report_test = classification_report(y_test, y_test_pred)
print("Classification Report of Training Set:")
print(report_train)
print("Classification Report of Test Set:")
print(report_test)
```

    Confusion Matrix of Training Set:
    [[25  0  0  0  0  0  0  0  0]
     [ 0 14  0  0  0  0  0  0  0]
     [ 0  0 11  0  0  0  0  0  0]
     [ 0  0  0  4  0  0  0  0  0]
     [ 0  0  0  0 17  0  0  0  0]
     [ 0  0  0  0  0  2  0  0  0]
     [ 0  0  0  0  0  0 12  0  0]
     [ 0  0  0  0  0  0  0  3  0]
     [ 0  0  0  0  0  0  0  0 28]]
    Confusion Matrix of Test Set:
    [[14  1  0  0  0  1  0  0  1  1]
     [ 0  6  0  0  0  1  0  0  0  0]
     [ 0  2  1  0  0  1  0  1  0  1]
     [ 1  0  0  0  0  1  0  0  0  0]
     [ 0  0  0  0  0  3  0  0  0  1]
     [ 4  0  2  0  0  6  0  0  0  1]
     [ 0  0  0  0  0  0  0  0  0  2]
     [ 0  0  2  0  0  0  0  4  1  0]
     [ 0  0  0  0  0  0  0  0  0  1]
     [ 5  0  2  0  0  0  0  0  1 10]]
    Accuracy of Training Set: 1.0
    Accuracy of Test Set: 0.5256410256410257
    Classification Report of Training Set:
                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00        25
               2       1.00      1.00      1.00        14
               3       1.00      1.00      1.00        11
               4       1.00      1.00      1.00         4
               6       1.00      1.00      1.00        17
               7       1.00      1.00      1.00         2
               8       1.00      1.00      1.00        12
               9       1.00      1.00      1.00         3
              10       1.00      1.00      1.00        28
    
        accuracy                           1.00       116
       macro avg       1.00      1.00      1.00       116
    weighted avg       1.00      1.00      1.00       116
    
    Classification Report of Test Set:
                  precision    recall  f1-score   support
    
               1       0.58      0.78      0.67        18
               2       0.67      0.86      0.75         7
               3       0.14      0.17      0.15         6
               4       0.00      0.00      0.00         2
               5       0.00      0.00      0.00         4
               6       0.46      0.46      0.46        13
               7       0.00      0.00      0.00         2
               8       0.80      0.57      0.67         7
               9       0.00      0.00      0.00         1
              10       0.59      0.56      0.57        18
    
        accuracy                           0.53        78
       macro avg       0.32      0.34      0.33        78
    weighted avg       0.49      0.53      0.50        78
    
    

    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
X_encoded = pd.get_dummies(X)

# Split the data into training and test sets,suite3
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Confusion matrix
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix of Training Set:")
print(confusion_matrix_train)
print("Confusion Matrix of Test Set:")
print(confusion_matrix_test)

# Classification accuracy
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy of Training Set:", accuracy_train)
print("Accuracy of Test Set:", accuracy_test)

# Precision, Recall, and F1 score
report_train = classification_report(y_train, y_train_pred)
report_test = classification_report(y_test, y_test_pred)
print("Classification Report of Training Set:")
print(report_train)
print("Classification Report of Test Set:")
print(report_test)
```

    Confusion Matrix of Training Set:
    [[28  1  0  0  0  2  0  1  0  1]
     [ 4  8  3  0  0  0  0  3  0  1]
     [ 3  1  3  0  0  0  0  2  0  5]
     [ 0  2  1  0  0  2  0  0  0  0]
     [ 0  1  0  0  0  0  0  0  0  0]
     [ 7  1  2  0  0  7  0  2  0  5]
     [ 3  0  0  0  0  0  0  0  0  0]
     [ 4  2  0  0  0  2  0  9  0  0]
     [ 0  0  1  0  0  0  0  0  0  3]
     [ 4  3  2  0  0  3  0  2  0 21]]
    Confusion Matrix of Test Set:
    [[7 0 0 0 0 1 0 1 1]
     [0 0 0 0 0 0 0 2 0]
     [1 1 0 0 0 1 0 0 0]
     [1 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 2]
     [2 0 1 0 0 0 0 1 2]
     [0 0 0 0 0 1 0 0 0]
     [1 0 1 0 0 0 0 0 0]
     [4 2 1 0 0 1 0 1 2]]
    Accuracy of Training Set: 0.49032258064516127
    Accuracy of Test Set: 0.23076923076923078
    Classification Report of Training Set:
                  precision    recall  f1-score   support
    
               1       0.53      0.85      0.65        33
               2       0.42      0.42      0.42        19
               3       0.25      0.21      0.23        14
               4       0.00      0.00      0.00         5
               5       0.00      0.00      0.00         1
               6       0.44      0.29      0.35        24
               7       0.00      0.00      0.00         3
               8       0.47      0.53      0.50        17
               9       0.00      0.00      0.00         4
              10       0.58      0.60      0.59        35
    
        accuracy                           0.49       155
       macro avg       0.27      0.29      0.27       155
    weighted avg       0.44      0.49      0.45       155
    
    Classification Report of Test Set:
                  precision    recall  f1-score   support
    
               1       0.41      0.70      0.52        10
               2       0.00      0.00      0.00         2
               3       0.00      0.00      0.00         3
               4       0.00      0.00      0.00         1
               5       0.00      0.00      0.00         3
               6       0.00      0.00      0.00         6
               7       0.00      0.00      0.00         1
               8       0.00      0.00      0.00         2
              10       0.29      0.18      0.22        11
    
        accuracy                           0.23        39
       macro avg       0.08      0.10      0.08        39
    weighted avg       0.19      0.23      0.20        39
    
    

    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
dt = DecisionTreeClassifier(random_state=42)

# Train the model
dt.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

# Confusion matrix
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix of Training Set:")
print(confusion_matrix_train)
print("Confusion Matrix of Test Set:")
print(confusion_matrix_test)

# Classification accuracy
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Accuracy of Training Set:", accuracy_train)
print("Accuracy of Test Set:", accuracy_test)

# Precision, Recall, and F1 score
report_train = classification_report(y_train, y_train_pred)
report_test = classification_report(y_test, y_test_pred)
print("Classification Report of Training Set:")
print(report_train)
print("Classification Report of Test Set:")
print(report_test)
```

    Confusion Matrix of Training Set:
    [[33  0  0  0  0  0  0  0  0  0]
     [ 0 19  0  0  0  0  0  0  0  0]
     [ 0  0 14  0  0  0  0  0  0  0]
     [ 0  0  0  5  0  0  0  0  0  0]
     [ 0  0  0  0  1  0  0  0  0  0]
     [ 0  0  0  0  0 24  0  0  0  0]
     [ 0  0  0  0  0  0  3  0  0  0]
     [ 0  0  0  0  0  0  0 17  0  0]
     [ 0  0  0  0  0  0  0  0  4  0]
     [ 0  0  0  0  0  0  0  0  0 35]]
    Confusion Matrix of Test Set:
    [[7 1 0 0 0 0 1 0 0 1]
     [0 1 0 0 0 1 0 0 0 0]
     [1 0 1 0 0 0 0 0 0 1]
     [0 0 0 1 0 0 0 0 0 0]
     [0 0 0 1 0 1 1 0 0 0]
     [1 0 0 1 0 2 0 0 0 2]
     [0 0 0 0 0 0 0 0 0 1]
     [0 0 1 0 0 0 0 1 0 0]
     [0 0 0 0 0 0 0 0 0 0]
     [3 0 1 0 0 1 0 0 1 5]]
    Accuracy of Training Set: 1.0
    Accuracy of Test Set: 0.46153846153846156
    Classification Report of Training Set:
                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00        33
               2       1.00      1.00      1.00        19
               3       1.00      1.00      1.00        14
               4       1.00      1.00      1.00         5
               5       1.00      1.00      1.00         1
               6       1.00      1.00      1.00        24
               7       1.00      1.00      1.00         3
               8       1.00      1.00      1.00        17
               9       1.00      1.00      1.00         4
              10       1.00      1.00      1.00        35
    
        accuracy                           1.00       155
       macro avg       1.00      1.00      1.00       155
    weighted avg       1.00      1.00      1.00       155
    
    Classification Report of Test Set:
                  precision    recall  f1-score   support
    
               1       0.58      0.70      0.64        10
               2       0.50      0.50      0.50         2
               3       0.33      0.33      0.33         3
               4       0.33      1.00      0.50         1
               5       0.00      0.00      0.00         3
               6       0.40      0.33      0.36         6
               7       0.00      0.00      0.00         1
               8       1.00      0.50      0.67         2
               9       0.00      0.00      0.00         0
              10       0.50      0.45      0.48        11
    
        accuracy                           0.46        39
       macro avg       0.36      0.38      0.35        39
    weighted avg       0.46      0.46      0.45        39
    
    

    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

### Task 3.3


```python

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Confusion Matrix
knn_cm = confusion_matrix(y_test, knn_pred, labels=np.unique(y_test))
dt_cm = confusion_matrix(y_test, dt_pred, labels=np.unique(y_test))

# Visualization the data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
pd.DataFrame(knn_cm, index=np.unique(y_test), columns=np.unique(y_test)).plot(
    kind='bar', cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix for K-Nearest Neighbors')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
pd.DataFrame(dt_cm, index=np.unique(y_test), columns=np.unique(y_test)).plot(
    kind='bar', cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix for Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('True')


plt.tight_layout()


print("K-Nearest Neighbors:")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print("Precision:", precision_score(y_test, knn_pred, average='weighted', zero_division=1))
print("Recall:", recall_score(y_test, knn_pred, average='weighted', zero_division=1))
print("F1 Score:", f1_score(y_test, knn_pred, average='weighted'))

print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred, average='weighted', zero_division=1))
print("Recall:", recall_score(y_test, dt_pred, average='weighted', zero_division=1))
print("F1 Score:", f1_score(y_test, dt_pred, average='weighted'))
plt.savefig('Figure 17')
plt.show()
# true refers to the true class labes and count is the number of ocuurences of predictions
```

    C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    

    K-Nearest Neighbors:
    Accuracy: 0.23076923076923078
    Precision: 0.3143719026071967
    Recall: 0.23076923076923078
    F1 Score: 0.1956315289648623
    
    Decision Tree:
    Accuracy: 0.5384615384615384
    Precision: 0.6180144641683104
    Recall: 0.5384615384615384
    F1 Score: 0.5256410256410257
    


    
![png](output_48_2.png)
    



```python

```
