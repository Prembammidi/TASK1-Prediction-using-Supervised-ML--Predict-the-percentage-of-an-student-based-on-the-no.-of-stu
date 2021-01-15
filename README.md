# TASK1-Prediction-using-Supervised-ML--Predict-the-percentage-of-an-student-based-on-the-no.-of-stu

BAMMIDI PREM KUMAR
DATA SCIENCE AND BUSINESS ANALYTICS
GRIP - THE SPARKS FOUNDATION
TASK1: Prediction using Supervised ML - Predict the percentage of an student based on the no. of study hours
In [3]:
# import all the libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline
In [8]:
# import data from link 

url = "http://bit.ly/w-data"
stu_df = pd.read_csv(url)

stu_df.head(20)
Out[8]:
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
10	7.7	85
11	5.9	62
12	4.5	41
13	3.3	42
14	1.1	17
15	8.9	95
16	2.5	30
17	1.9	24
18	6.1	67
19	7.4	69
In [10]:
# scores plotting distribution

stu_df.plot(x='Hours', y='Scores', style='o')
plt.title('H vs P')
plt.xlabel('hours spent for study')
plt.ylabel('percentage scored')
plt.show()

In [12]:
x = s_data.iloc[:, :-1].values
y = s_data.iloc[:, 1].values
In [16]:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
In [24]:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("complete")
complete
In [26]:
line = regressor.coef_*x+regressor.intercept_

plt.scatter(x, y)
plt.plot(x, line);
plt.show()

Predictions
In [27]:
print(x_test)
y_pred = regressor.predict(x_test)
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
In [28]:
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
Out[28]:
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
In [29]:
new_x = [[9.25]]
new_pred = regressor.predict(new_x)
print("No of Hours = {}".format(new_x))
print("predcition score = {}".format(new_pred[0]))
No of Hours = [[9.25]]
predcition score = 93.69173248737538
Evalution
In [33]:
from sklearn import metrics
print("Mean abs error",
     metrics.mean_absolute_error(y_test, y_pred))
Mean abs error 4.183859899002975
