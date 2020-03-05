# Spam detector for e-mails and SMS messages

~~ *This project was implemented in March 2017* ~~

The aim of this project was to find the best spam detector by comparing supervised classification methods on SMS and email data.

## Use of the Python 3 library *scikit-learn*
A tutorial on text classification with *scikit-learn* can be found [**here**](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

## Data, labels and 
The class we are looking for is "spam", this is the Class 1, the positive class. While class 0 is the negative class when hams are found.

There are 2 databases:
* one for SMS
* one for emails

The experimental parameters for learning are as follows:
- 5 cross-validation folds
- all SMS are randomly divided into 30% for training and 70% for testing
- all supervised emails are randomly divided into 1/3 for training and 2/3 for testing

The types of classifiers compared are : 
- Decision Trees
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayesians
By tuning them on their different parameters, we end up with 94 different classifiers to compare.

Several scoring functions are used to evaluate the classifiers:
- the proportion of well-classified examples
- the number of well-detected spam messages 
- the number of well-detected non-spams (hams) 
- the average precision (*accuracy*)
- the area under the ROC curve
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

La base d’emails est séparée en deux dossiers : Train et Test. Le dossier Train est lui même
découpé en trois dossiers contenant les exemples d’apprentissage : spam, ham et ham difficile.
Chaque fichier contient le code source d’un email. Les fichiers sont donc classés selon le type de
message qu’ils contiennent puisque le nom du dossier dans lequel ils sont représente le nom du
label supervisé.

seulement faire de l’apprentissage sur le dossier Train (l’autre
dossier n’étant pas classé). Le dossier contient 1720 emails au total, dont 1376 hams et 344 spams.
Les deux classes ne sont donc pas équilibrées.
Une fois dans ce dossier, pour charger en mémoire les fichiers textes avec comme catégorie le nom
du sous-dossier spam, easy_ham ou hard_ham

Comme pour les SMS, la classe 1 (positive) est spam. J’ai choisi de regrouper easy_ham et
hard_ham afin de n’avoir qu’une autre classe 0 (négative) représentant les hams en général.

deux tiers de Train pour l’ensemble d’apprentissage, et un tiers pour
l’ensemble de test.

L’exécution de la classification pour la base des emails se lance avec la commande
suivante :
python3 classification_Email.py
A la suite de cette éxécution, j’obtiens un fichier de résultats par mesure de performance
sur les 94 classifieurs pour la base des emails. 

## Results

Finally, the best detector turns out to be an SVM with 98% accuracy and 96% area under the ROC curve. 
This SVM can be used via the python script *SpamDetector.py*.

For more details and a full report of the experimental results, please refer to the file *Rapport.pdf*, which is written in French.
