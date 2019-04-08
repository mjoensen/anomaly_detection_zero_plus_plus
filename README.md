## Zero++ Anomaly Detection Algorithm
This repository contains an naive implementation of the "Zero++" algorithm in Python3 to better understand the algorithm. The algorithm is from the paper: **ZERO++: Harnessing the Power of Zero Appearances to Detect Anomalies in Large-Scale Data Sets** from 2016 by Pang, Ting, Albrecht and Jin. **All credit goes to them**. I did not in any way invent the algorithm or had any part in producing the paper.

This repository was created as the only implementation I was able to find of this interesting algorithm was the authors own implementation in the WEKA-framework in Java. I found this implementation very unreadable and decided to implement and publish a more readable version for the benefit of other interested parties. I do not in any way guarantee that the implementation is correct but I get very similar results to the ones described in the paper.  

The implementation is quite naive, as I tried to follow the algorithm as close as possible. I do however store the "probability tables" described in the paper in another fashion as I found it easier to work with. The implementation only works with pure categorical data.

The implementation is done in Python 3, specifically in Python 3.6.7. I use additional libraries such as *pandas* and *numpy*. This can be installed by utilizing the *requirements.txt*-file and the following command:
```
pip3 install -r requirements.txt
```
The implementation loads the classic Mushroom-dataset from UCI Machine Learning Repository, retains only 5% of the poisonous class and one-hot encodes the data and computes the anomaly score for each row in the dataset. The implementation can be run using the command:
```
python3 zero++.py
```
This outputs an anomaly score for each row and the label for the row.
