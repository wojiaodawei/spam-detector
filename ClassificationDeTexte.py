#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David Condaminet (21306226)"

"""
    Ceci est le module permettant de faire de la classification
     à partir d'une base de données de textes en utilisant les principaux
     classifieurs et en testant différents jeux de paramètres sur chacun d'entre eux.
"""

import numpy as np
from sklearn.metrics import *
from sklearn import svm, neighbors, tree
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import Metrics

def fusionne2Dicts(x, y):
    """
        Fusionne deux dictionnaires.
    """
    z = x.copy()
    z.update(y)
    return z

def renommeClefsDictClf(d):
    """
        Renomme les clefs d'un dictionnaire d'un classifieur
        en leur ajoutant le préfixe «clf__».
    """
    clefs = list(d.keys())
    for cle in clefs:
        nouvelle_cle = 'clf__' + cle
        d[nouvelle_cle] = d.pop(cle)
    return d

#
class DenseTransformer(TfidfTransformer):
    """
        Descripteur qui permet de transformer le texte en matrice.
    """

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

class ClassificationDeTexte():
    """
        Classe représentant la classification générale d'une base de données
        en calculant les scores de différents classifieurs sur différentes mesures de performance.
    """

    def __init__(self, type_textes, X, y, taille_test):
        self.type_textes = type_textes
        self.X = X
        self.y = y
        self.cv = ShuffleSplit(n_splits=5, random_state=42, test_size=taille_test)

    def classifieurDecisionTree(self):
        """
            Définit le classifieur Arbre de décision et son ensemble de jeux de paramètres
        """
        tuned_parameters = {'criterion': ['gini', 'entropy']}
        return tree.DecisionTreeClassifier(), tuned_parameters

    def classifieurSVC(self):
        """
            Définit le classifieur SVM SVC et son ensemble de jeux de paramètres
        """
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                             'C': [1e-2, 1e-1, 1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1e-2, 1e-1, 1, 10, 100, 1000]},
                            {'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5], 'C': [1, 2, 3, 4, 5]}]
        return svm.SVC(), tuned_parameters

    def classifieurLinearSVC(self):
        """
            Définit le classifieur SVM LinearSVC et son ensemble de jeux de paramètres
        """
        tuned_parameters = {'C': [1e-2, 1e-1, 1, 10, 100, 1000]}
        return svm.LinearSVC(), tuned_parameters

    def classifieurKPPV(self):
        """
            Définit le classifieur k-Plus-Proches-Voisins et son ensemble de jeux de paramètres
        """
        tuned_parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
                             'weights': ['uniform', 'distance']}
        return neighbors.KNeighborsClassifier(), tuned_parameters

    def classifieurGaussianNaiveBayes(self):
        """
            Définit le classifieur Gaussian Naive Bayes et son ensemble de jeux de paramètres
        """
        tuned_parameters = {}
        return GaussianNB(), tuned_parameters

    def classifieurMultinomialNaiveBayes(self):
        """
            Définit le classifieur Multinomial Naive Bayes et son ensemble de jeux de paramètres
        """
        tuned_parameters = {}
        return MultinomialNB(), tuned_parameters

    def classifieurBernoulliNaiveBayes(self):
        """
            Définit le classifieur Bernoulli Naive Bayes et son ensemble de jeux de paramètres
        """
        tuned_parameters = {}
        return BernoulliNB(), tuned_parameters

    def compareClassifieurs(self, mesure):
        """
            Lance la validation croisée pour chaque classifieur en fonction d'une mesure donnée
        """
        scores = {}
        classifieurs = {'Arbre de Décision': self.classifieurDecisionTree(),
                        'SVC': self.classifieurSVC(),
                        'LinearSVC': self.classifieurLinearSVC(),
                        'kPPV': self.classifieurKPPV(),
                        'Gaussian Naive Bayes': self.classifieurGaussianNaiveBayes(),
                        'Multinomial Naive Bayes': self.classifieurMultinomialNaiveBayes(),
                        'Bernoulli Naive Bayes': self.classifieurBernoulliNaiveBayes()}
        for nom_clf in classifieurs:
            clf, clf_tuned_parameters = classifieurs[nom_clf]
            if nom_clf == "Gaussian Naive Bayes":
                text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                                     ('dstf', DenseTransformer()), ('clf', clf)])
            else:
                text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                                     ('clf', clf)])
            scores = self.validationCroisee(text_clf, clf_tuned_parameters, nom_clf, mesure, scores)
        return scores

    # Cross-validation des paramètres de l'algorithme d'apprentissage
    def validationCroisee(self, clf, clf_tuned_parameters, nom_clf, mesure, dico_scores):
        """
            Fais la validation croisée sur un classifieur donné
        """
        # pipeline_tuned_parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False)}
        pipeline_tuned_parameters = {}
        tuned_parameters = None
        # si c'est une liste de dicts
        if isinstance(clf_tuned_parameters, list):
            tuned_parameters = []
            for d in clf_tuned_parameters:
                renommeClefsDictClf(d)
                tuned_parameters += [fusionne2Dicts(pipeline_tuned_parameters, d)]
        elif isinstance(clf_tuned_parameters, dict):
            clf_tuned_parameters = renommeClefsDictClf(clf_tuned_parameters)
            tuned_parameters = fusionne2Dicts(pipeline_tuned_parameters, clf_tuned_parameters)
        clf_grid = GridSearchCV(clf, tuned_parameters, cv=self.cv, scoring=mesure, n_jobs=-1, verbose=True)
        clf_grid.fit(self.X, self.y)
        elements = clf_grid.cv_results_
        for i in np.argsort(elements['mean_test_score']):
            score = elements['mean_test_score'][i]
            sortie = "%0.3f (+/-%0.03f) pour : %s %r"% (score,
                                               elements['std_test_score'][i],
                                               nom_clf,
                                               elements['params'][i])
            print(sortie)
            dico_scores[sortie] = score
        return dico_scores

    def triResultats(self, d):
        """
            Tri le dictionnaire de tous les résultats par ordre décroissant sur le score
        """
        d = d.items()
        d = sorted(d, key=lambda score: score[1], reverse=True)
        return d

    def enregistre(self, fichier, liste):
        """
            Enregistre dans un fichier tous les résultats correctement triés
        """
        for score in liste:
            fichier.write(score[0] + "\n")
        fichier.close()

    def classification(self):
        """
            Méthode principale lançant la comparaison de tous les classifieurs définis
            sur toutes les mesures de performance définies
        """
        print("Pourcentage Exemples Bien Classes")
        f1 = open("resultats/" + self.type_textes + "/pourcentage_exemples_bien_classes.txt", "w")
        results1 = self.compareClassifieurs(make_scorer(Metrics.pourcentageExemplesBienClasses))
        results1 = self.triResultats(results1)
        self.enregistre(f1, results1)

        print("\nNombre Spams Bien Detectes")
        f2 = open("resultats/" + self.type_textes + "/nb_spams_bien_detectes.txt", "w")
        results2 = self.compareClassifieurs(make_scorer(Metrics.nbSpamsBienDetectes))
        results2 = self.triResultats(results2)
        self.enregistre(f2, results2)

        print("\nNombre Hams Bien Detectes")
        f3 = open("resultats/" + self.type_textes + "/nb_hams_bien_detectes.txt", "w")
        results3 = self.compareClassifieurs(make_scorer(Metrics.nbHamsBienDetectes))
        results3 = self.triResultats(results3)
        self.enregistre(f3, results3)

        print("\nPrecision Moyenne")
        f4 = open("resultats/" + self.type_textes + "/precision_moyenne.txt", "w")
        results4 = self.compareClassifieurs(make_scorer(Metrics.precisionMoyenne))
        results4 = self.triResultats(results4)
        self.enregistre(f4, results4)

        print("\nAire Sous La Courbe ROC")
        f5 = open("resultats/" + self.type_textes + "/aire_sous_la_courbe_ROC.txt", "w")
        results5 = self.compareClassifieurs(make_scorer(Metrics.aireSousLaCourbeROC))
        results5 = self.triResultats(results5)
        self.enregistre(f5, results5)


