# Covid Explorer

Exploration des data de l'évolution de l'épidémie de Coronavirus.

Les datas sont mises à disposition par le Johns Hopkins Coronavirus Resource Center et comptabilise les contamnisations journalières cumulées du 22 janvier 2020 au 10 avril 2021.

Site disponible à l'adresse suivante :
https://share.streamlit.io/mickaelkohler/covid_explorer/main/covid_explorer.py


## Sommaire

* [Origine du projet](#origine-du-projet)
* [Screenshots](#interface)
* [Technologies](#technologies)
* [Bases de Données](#bases-de-données)
* [Statut](#statut)
* [La Team](#la-team)

## Origine du projet

La **WebApp** crée se divise en 3 sections : 
- Une exploration de l'évolution de l'épidémie jsuqu'au 10/04/2021
- Une comparaison de deux modèles de prédictions des contamnisations par statsmodels (ARIMA) et PROPHET
- Automatisation d'un modèle de prédiction des cas futures à partir du 10/04/2021

## Interface

Ces analyses ont été mise à disposition au travers d’une __WebApp__ créée au travers de la plateforme __Streamlit__.

## Technologies 

Projet fait entièrement en **Python**

Utilisations des librairies suivantes : 
 - Pandas
 - statsmodels
 - Plotly
 - Streamlit
 - prophet

## Bases de données 

La [base de données de **Johns Hopkins Coronavirus Resource Center**](https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases) ont été utilisées pour obtenir l’ensemble des datas sur l'épidémie.

## La Team

Le projet a été réalisé par les élèves de la **Wild Code School** : 
- Franck Loiselet
- Franck Maillet
- Michael Kohler
