# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Author: Oliver Poole
Model for the Udacity Machine Learning DevOps Nano Degree
date. 04 January 2022
Model Implemented: Random Forest Classifier

## Intended Use
Primary intented use:
This model's primary use is to build a continuous delivery pipeline around it for the Udacity Machine Learning DevOps Nano Degree.
Therefore the performance is not the primary concern of the model.

## Training Data
The training data is the census.csv from the publicly available Census Bureau data.

80% of the data was used for training and 20% was used for testing using the train test split method from scikit learn. No random seed or stratification was used.

## Evaluation Data
The model is evaluated on the test set. The 

## Metrics
Precision, recall and the fbeta score are used for evaluating teh metrics on different data slices.

overall model performance :
precision: 0.8096707818930041
recall: 0.4987325728770596
fbeta: 0.6172549019607844

Category  Bachelors has precision: 0.7296, recall: 0.8212516884286357, fbeta: 0.772717644566829
Category  HS-grad has precision: 0.9871244635193133, recall: 0.1373134328358209, fbeta: 0.2410901467505241
Category  11th has precision: 1.0, recall: 0.13333333333333333, fbeta: 0.23529411764705882
Category  Masters has precision: 0.8009708737864077, recall: 0.8602711157455682, fbeta: 0.8295625942684766
Category  9th has precision: 1.0, recall: 0.037037037037037035, fbeta: 0.07142857142857142
Category  Some-college has precision: 0.9832635983263598, recall: 0.16943042537851477, fbeta: 0.28905289052890526
Category  Assoc-acdm has precision: 1.0, recall: 0.1811320754716981, fbeta: 0.30670926517571884
Category  Assoc-voc has precision: 0.9722222222222222, recall: 0.19390581717451524, fbeta: 0.3233256351039261
Category  7th-8th has precision: 1.0, recall: 0.1, fbeta: 0.18181818181818182
Category  Doctorate has precision: 0.8503401360544217, recall: 0.8169934640522876, fbeta: 0.8333333333333333
Category  Prof-school has precision: 0.8691588785046729, recall: 0.8794326241134752, fbeta: 0.8742655699177438
Category  5th-6th has precision: 1.0, recall: 0.0625, fbeta: 0.11764705882352941
Category  10th has precision: 1.0, recall: 0.08064516129032258, fbeta: 0.1492537313432836
Category  1st-4th has precision: 1.0, recall: 0.16666666666666666, fbeta: 0.2857142857142857
Category  Preschool has precision: 1.0, recall: 1.0, fbeta: 1.0
Category  12th has precision: 1.0, recall: 0.21212121212121213, fbeta: 0.35

## Ethical Considerations
The data here is strictly anonymous and cannot be traced back to the original person. The data is publicly available.

## Caveats and Recommendations
The model here is not particularly powerful and many improvements can be made to make it more powerful. This needs considerable time and resource investment and is therefore out of teh scope for this project.