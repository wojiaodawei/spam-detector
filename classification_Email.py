#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David Condaminet (21306226)"

"""
    Ceci est le module paramétrant la classification de textes
    pour la base de données des Emails (avec 1/3 de la base pour le test).
"""

from sklearn.datasets import load_files
import ClassificationDeTexte


def nettoieMail(email):
    """
        Nettoie l'email en supprimant son entête
    """
    lignes_email = email.split(b"\n")
    i = 0
    while i < len(lignes_email) and lignes_email[i] != b'':
        i += 1
    email = b"\n".join(lignes_email[i + 1:])
    return email.decode('utf-8', 'ignore')

def creationBase(chemin_container):
    """
        Crée la base de données en extrayant les données et les classes
    """
    train = load_files(container_path=chemin_container, random_state=42)
    # print(len(train.data))
    # print(train.target_names)
    # print(train.filenames[2])
    # print(nettoieMail(train.data[0]))
    X = []
    y = []
    for i in range(len(train.data)):
        if ".DS_Store" not in train.filenames[i]:
            email = train.data[i]
            email = nettoieMail(email)
            X += [email]
            classe = train.target_names[train.target[i]]
            y += [int(classe == 'spam')]
    # Taille de la BDD
    # print(len(X), len(y))
    # # Nombre d'attributs de chaque exemple (dimensions des descripteurs)
    # print(X[2])
    # print(y[2])
    return X, y

if __name__ == '__main__':
    X, y = creationBase("Base_Email/train")
    text_clf = ClassificationDeTexte.ClassificationDeTexte("Email", X, y, 0.33)
    text_clf.classification()