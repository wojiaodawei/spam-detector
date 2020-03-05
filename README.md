# Spam detector for e-mails and SMS messages

~~ *This project was implemented in March 2017* ~~

The aim of this project was to find the best spam detector by comparing supervised classification methods on SMS and email data.

The data folder is missing. Only source code is available here.

## Use of the Python 3 library *scikit-learn*
A tutorial on text classification with *scikit-learn* can be found [**here**](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).

## Experimental details to know

There are 2 databases:
* one for SMS
* one for emails

"spam" is the Class 1, the positive class. While class 0 is the negative class "ham".

The experimental parameters for learning are as follows:
* 5 cross-validation folds
* all SMS are randomly divided into 30% for training and 70% for testing
* all supervised emails are randomly divided into 2/3 for training and 1/3 for testing

The types of classifiers compared are : 
* Decision Trees
* Support Vector Machine
* K-Nearest Neighbors
* Naive Bayesians
By tuning them on their different parameters, we end up with 94 different classifiers to compare.

Several scoring functions are used to evaluate the classifiers:
* the proportion of well-classified examples
* the number of well-detected spam messages 
* the number of well-detected non-spams (hams) 
* the average precision (*accuracy*)
* the area under the ROC curve
These 5 evaluation criteria are defined in the module *Metrics.py*.

The results are saved in the directory *results/*.

## SMS

In the text file *SMSSpamCollection.txt*, the SMS messages are represented line by line, by the type of message (*spam* or *ham*) and the message itself, both separated by a tab. 
In this database, there are 5574 SMS and among these SMS, 4827 are hams and 747 are spam. So the two classes are not well balanced.

Execute the classification of the SMS with all these criteria and all these classifiers by typing the command:
```
python3 classification_SMS.py
```

## Email

The email database is split into two folders: *Train* and *Test*. The *Train* folder is itself
divided into three folders containing learning examples: *spam*, *ham* and *hard ham*.
Each file contains the source code of an email. 

Learning is only done on the *Train* folder (the other folder is not classified/supervised). 
The folder contains a total of 1720 emails, including 1376 hams and 344 spams. The two classes are unbalanced.

As for SMS, class 1 (positive) is *spam*. *easy_ham* and *hard_ham* are merged to have only one other class 0 (negative) representing *hams* in general.

Execute the classification for the email database by typing the following command:
```
python3 classification_Email.py
```

## Results

Finally, the best detector turns out to be an SVM with 98% accuracy and 96% area under the ROC curve. 
This SVM can be used via the python script *SpamDetector.py*.

For more details and a full report of the experimental results, please refer to the file *Rapport.pdf*, which is written in French.
