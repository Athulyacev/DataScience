

import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

data=pd.read_csv('iris.csv')
data.head()

	5.1 	3.5 	1.4 	0.2 	Iris-setosa
0 	4.9 	3.0 	1.4 	0.2 	Iris-setosa
1 	4.7 	3.2 	1.3 	0.2 	Iris-setosa
2 	4.6 	3.1 	1.5 	0.2 	Iris-setosa
3 	5.0 	3.6 	1.4 	0.2 	Iris-setosa
4 	5.4 	3.9 	1.7 	0.4 	Iris-setosa

x=data.iloc[:,:4]
x.head()

	5.1 	3.5 	1.4 	0.2
0 	4.9 	3.0 	1.4 	0.2
1 	4.7 	3.2 	1.3 	0.2
2 	4.6 	3.1 	1.5 	0.2
3 	5.0 	3.6 	1.4 	0.2
4 	5.4 	3.9 	1.7 	0.4

y=data.iloc[:,-1]
y.head()

0    Iris-setosa
1    Iris-setosa
2    Iris-setosa
3    Iris-setosa
4    Iris-setosa
Name: Iris-setosa, dtype: object

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
x_train

	5.1 	3.5 	1.4 	0.2
109 	6.5 	3.2 	5.1 	2.0
90 	6.1 	3.0 	4.6 	1.4
138 	6.9 	3.1 	5.4 	2.1
4 	5.4 	3.9 	1.7 	0.4
129 	7.4 	2.8 	6.1 	1.9
... 	... 	... 	... 	...
130 	7.9 	3.8 	6.4 	2.0
128 	7.2 	3.0 	5.8 	1.6
111 	6.8 	3.0 	5.5 	2.1
87 	5.6 	3.0 	4.1 	1.3
56 	4.9 	2.4 	3.3 	1.0

104 rows × 4 columns

x_test

	5.1 	3.5 	1.4 	0.2
122 	6.3 	2.7 	4.9 	1.8
120 	5.6 	2.8 	4.9 	2.0
16 	5.1 	3.5 	1.4 	0.3
8 	4.9 	3.1 	1.5 	0.1
94 	5.7 	3.0 	4.2 	1.2
125 	6.2 	2.8 	4.8 	1.8
78 	5.7 	2.6 	3.5 	1.0
46 	4.6 	3.2 	1.4 	0.2
82 	6.0 	2.7 	5.1 	1.6
145 	6.3 	2.5 	5.0 	1.9
134 	7.7 	3.0 	6.1 	2.3
49 	7.0 	3.2 	4.7 	1.4
28 	4.7 	3.2 	1.6 	0.2
147 	6.2 	3.4 	5.4 	2.3
15 	5.4 	3.9 	1.3 	0.4
41 	4.4 	3.2 	1.3 	0.2
148 	5.9 	3.0 	5.1 	1.8
143 	6.7 	3.3 	5.7 	2.5
64 	6.7 	3.1 	4.4 	1.4
146 	6.5 	3.0 	5.2 	2.0
71 	6.3 	2.5 	4.9 	1.5
40 	4.5 	2.3 	1.3 	0.3
113 	5.8 	2.8 	5.1 	2.4
22 	5.1 	3.3 	1.7 	0.5
126 	6.1 	3.0 	4.9 	1.8
5 	4.6 	3.4 	1.4 	0.3
59 	5.0 	2.0 	3.5 	1.0
63 	5.6 	2.9 	3.6 	1.3
124 	7.2 	3.2 	6.0 	1.8
100 	5.8 	2.7 	5.1 	1.9
48 	5.0 	3.3 	1.4 	0.2
75 	6.8 	2.8 	4.8 	1.4
68 	5.6 	2.5 	3.9 	1.1
23 	4.8 	3.4 	1.9 	0.2
61 	6.0 	2.2 	4.0 	1.0
141 	5.8 	2.7 	5.1 	1.9
88 	5.5 	2.5 	4.0 	1.3
121 	7.7 	2.8 	6.7 	2.0
35 	5.5 	3.5 	1.3 	0.2
0 	4.9 	3.0 	1.4 	0.2
62 	6.1 	2.9 	4.7 	1.4
19 	5.4 	3.4 	1.7 	0.2
86 	6.3 	2.3 	4.4 	1.3
36 	4.9 	3.1 	1.5 	0.1
80 	5.5 	2.4 	3.7 	1.0

sc=StandardScaler()
sc.fit(x_train)
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)

classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)

KNeighborsClassifier()

y_pred=classifier.predict(x_test)
y_pred

C:\Users\hp\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
  mode, _ = stats.mode(_y[neigh_ind, k], axis=1)

array(['Iris-virginica', 'Iris-versicolor', 'Iris-setosa', 'Iris-setosa',
       'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor',
       'Iris-setosa', 'Iris-virginica', 'Iris-virginica',
       'Iris-virginica', 'Iris-versicolor', 'Iris-setosa',
       'Iris-virginica', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor',
       'Iris-virginica', 'Iris-versicolor', 'Iris-virginica',
       'Iris-virginica', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa',
       'Iris-virginica', 'Iris-setosa', 'Iris-versicolor',
       'Iris-versicolor', 'Iris-virginica', 'Iris-virginica',
       'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa',
       'Iris-versicolor', 'Iris-virginica', 'Iris-versicolor',
       'Iris-virginica', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor',
       'Iris-setosa', 'Iris-versicolor', 'Iris-setosa', 'Iris-versicolor'],
      dtype=object)

y_test

122     Iris-virginica
120     Iris-virginica
16         Iris-setosa
8          Iris-setosa
94     Iris-versicolor
125     Iris-virginica
78     Iris-versicolor
46         Iris-setosa
82     Iris-versicolor
145     Iris-virginica
134     Iris-virginica
49     Iris-versicolor
28         Iris-setosa
147     Iris-virginica
15         Iris-setosa
41         Iris-setosa
148     Iris-virginica
143     Iris-virginica
64     Iris-versicolor
146     Iris-virginica
71     Iris-versicolor
40         Iris-setosa
113     Iris-virginica
22         Iris-setosa
126     Iris-virginica
5          Iris-setosa
59     Iris-versicolor
63     Iris-versicolor
124     Iris-virginica
100     Iris-virginica
48         Iris-setosa
75     Iris-versicolor
68     Iris-versicolor
23         Iris-setosa
61     Iris-versicolor
141     Iris-virginica
88     Iris-versicolor
121     Iris-virginica
35         Iris-setosa
0          Iris-setosa
62     Iris-versicolor
19         Iris-setosa
86     Iris-versicolor
36         Iris-setosa
80     Iris-versicolor
Name: Iris-setosa, dtype: object

cm=confusion_matrix(y_test,y_pred)
cm

array([[15,  0,  0],
       [ 0, 13,  2],
       [ 0,  2, 13]], dtype=int64)

ac=accuracy_score(y_test,y_pred)
ac

0.9111111111111111

 

