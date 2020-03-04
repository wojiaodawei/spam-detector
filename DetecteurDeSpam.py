#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David Condaminet (21306226)"

"""
    Ceci est le module appliquant le meilleur classifieur de «classification_Email»
    aux données du répertoire Test qui n'est pas classé.
"""

import classification_Email
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm
import os

class DetecteurDeSpamEmail():
    """
        Classe représentant le détecteur de spams pour les emails
        qui prend en entrée le «meilleur classifieur» résultant de
        la classification de «classification_Email»
    """

    def __init__(self, meilleur_clf):
        self.meilleur_clf = meilleur_clf
        self.X, self.y = classification_Email.creationBase("Base_Email/train")
        self.action()
        self.enregistre()

    def action(self):
        """
            Le classifieur donné apprend sur toute la base Train
            et prédit ensuite la classe de la base Test
        """
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                             ('clf', self.meilleur_clf)])
        text_clf.fit(self.X, self.y)

        X_test = []
        self.filenames_test = []
        for fichier in os.listdir('Base_Email/test'):
            if not(os.path.isdir(fichier)) and fichier != ".DS_Store":
                self.filenames_test += [fichier]
                with open("Base_Email/test/" + fichier, 'rb') as f:
                    email = f.read()
                    email = classification_Email.nettoieMail(email)
                    X_test += [email]
        self.y_test = text_clf.predict(X_test)

    def enregistre(self):
        """
            Enregistre la prédiction dans un fichier «resultats_TEST_Condaminet.txt»
        """
        fichier_sortie = open("resultats_TEST_Condaminet.txt", "w")
        for i in range(len(self.y_test)):
            y = self.y_test[i]
            filename = self.filenames_test[i]
            fichier_sortie.write(filename + " " + str(y) + "\n")
        fichier_sortie.close()

if __name__ == '__main__':
    DetecteurDeSpamEmail(svm.SVC(C=1000, gamma=0.01, kernel='rbf'))