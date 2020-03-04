# Spam detector for e-mails and SMS messages

~~ *This project was implemented in March 2017* ~~

L'objectif est de trouver le meilleur détecteurs de spams en comparant des méthodes de classification supervisée sur les données des
bases de SMS et d’emails.
utilise la librairie scikit-learn sur Python 3

tutoriel sur la classification de textes
La tutoriel disponible à la page http://scikit-learn.org/stable/tutorial/text_
analytics/working_with_text_data.html présente des outils de scikit-learn dont j’ai eu besoin
pour réaliser ce projet.

La classe recherchée est donc
«spam». La classe 1 est la classe positive où se trouvent les spams recherchés, tandis que la classe
0 est la classe négative où se trouvent les hams.
On dispose de deux bases de données :
— une pour les SMS
— une pour les emails

Sous forme d’un fichier texte SMSSpamCollection.txt, les SMS sont représentés ligne par
ligne, par le type de message (spam ou ham) et le message lui-même, séparés tous deux par une
tabulation. Dans cette base, il y a 5574 SMS et parmi ces SMS, 4827 sont des hams et 747 sont
des spams. Les deux classes ne sont donc pas équilibrées.

5 validations croisées, l’ensemble de SMS sera divisé aléatoirement en
30% pour l’entraînement et 70% pour le test.

Types de classifieurs comparés : arbres de décision, SVM, kPPV, naïf bayésiens
En jouant sur leurs différents paramètres

plusieurs critères d’évaluation :
— la proportion d’exemples bien classés
— le nombre de spams bien détectés
— le nombre de non spams (hams) bien détectés
— la précision moyenne
— l’aire sous la courbe ROC
Ces cinq fonctions de scoring sont définies dans le module Metrics.py

Pour lancer la classification des SMS avec tous ces critères et tous ces classifieurs, il faut
taper la commande :
python3 classification_SMS.py

Les résultats sont sauvegardés dans le répertoire resultats.



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
sur les 94 classifieurs pour la base des emails. Tous ceux-ci sont enregistrés dans le répertoire
resultats/.


Au final, le meilleur détecteur s'avère être un SVM avec une précision de 98% et une aire sous la courbe ROC à 96%, celui-ci peut-être utilisé via le script python DetecteurDeSpam.py

Pour plus de détails et pour un compte-rendu des résultats expérimentaux, reportez-vous au fichier Rapport.pdf qui est écrit en français.
