#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David Condaminet (21306226)"

"""
    Ceci est le module paramétrant la classification de textes
    pour la base de données des SMS (avec 70% de la base pour le test).
"""

import ClassificationDeTexte


def creationBase(chemin_fichier_sms):
    """
        Crée la base de données en extrayant les données et les classes
    """
    fichier_sms = open(chemin_fichier_sms, 'rb')
    messages = fichier_sms.readlines()
    X = []
    y = []
    for message in messages:
        (classe, sms) = message.split(b'\t')
        X += [sms.decode('utf-8').strip()]
        y += [int(classe == b'spam')]
    # Taille de la BDD
    # print(len(X))
    # # Nombre d'attributs de chaque exemple (dimensions des descripteurs)
    # print(X[2])
    # print(y[2])
    return X, y

if __name__ == '__main__':
    X, y = creationBase('SMS_Spam/SMSSpamCollection.txt')
    text_clf = ClassificationDeTexte.ClassificationDeTexte("SMS", X, y, 0.7)
    text_clf.classification()

