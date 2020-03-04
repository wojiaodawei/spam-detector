#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "David Condaminet (21306226)"

"""
    Ceci est le module regroupant toutes les fonctions de score
     qu'utilise le module «ClassificationDeTexte».
"""

from sklearn.metrics import *
import numpy as np

def pourcentageExemplesBienClasses(y_vrai, y_pred):
    """
        Calcule le nombre d'exemples bien classés sur le nombre d'exemples total
    """
    mc = confusion_matrix(y_vrai, y_pred)
    res = (mc[0, 0] + mc[1, 1]) / np.sum(mc[:, :])
    return res

def nbSpamsBienDetectes(y_vrai, y_pred):
    """
        Calcule le nombre de spams bien détectés
    """
    mc = confusion_matrix(y_vrai, y_pred)
    res = mc[1, 1]
    return res

def nbHamsBienDetectes(y_vrai, y_pred):
    """
        Calcule le nombre de hams bien détectés
    """
    mc = confusion_matrix(y_vrai, y_pred)
    res = mc[0, 0]
    return res

def precisionMoyenne(y_vrai, y_pred):
    """
        Calcule la précision moyenne c'est-à-dire l'aire sous la courbe Rappel/Précision
    """
    return average_precision_score(y_vrai, y_pred)

def aireSousLaCourbeROC(y_vrai, y_pred):
    """
        Calcule l'aire sous la courbe ROC
    """
    return roc_auc_score(y_vrai, y_pred)

