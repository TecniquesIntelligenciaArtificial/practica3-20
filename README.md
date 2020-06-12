# Pràctica 3. Classificació

### Dataset
Usarem el data set wine.csv que teniu al gitHub. Podeu trobar una descripció del dataset a 
https://archive.ics.uci.edu/ml/datasets/wine
És un dataset prou "amable" que es deixa classificar prou bé

### Llibreries de classificació
Farem servir el scikit-learn  que és una llibreria molt usada per anàlisi de dades i datamining. És una llibreria 
professional que competeix amb el R. Està documentat que scikit-learn és més ràpid que R. Com sempre hi ha gent
que prefereix R i d'altres que prefereixen scikit-learn i també n'hi ha que diuen que s'han d'usar els dos entonrs que
on no arriba un arriba l'altre. Per questa pràctica hem escollit scikit-learn perquè ja coneixeu python, tot i que no
és una pràctica de programar gaire.

Per usar scikit-learn l'haureu d'instal·lar. Aquí trobareu documentació de com fer-ho 
https://scikit-learn.org/stable/index.html

### Objectiu
Al fitxer *classifyWine.py* trobareu la pràctica quasi feta. L'objectiu és que feu una darrera pràctica senzilla que 
us permeti veure com es treballa amb la llibreria i que compareu els tres algorismes de classificació que hem vists a classe: K-nearest
neighbors, decision trees i el naive bayes. 

Un altre objectiu és que vegueu que usar les dades tal com venen (raw data) no sempre és una bona idea. Moltes vegades
(diria que sempre) s'han de preprocessar. De fet en un projecte real de data minig el 80% del temps és destina a conéixer
i preprocessar les dades ("la famosa cuina") i el 20% restant a aplicar els diveros algorismes d'aprenentatge  

En aquesta pràctica usarem tres preprocessats de les dades diferents:
* El buit (None): és a dir no fer res i usar les dades tal com venen (raw data)
* Normalitzant els attributs: molts algorismes són depenents del rang dels atributs. Per exemple, en el càlcul d'una 
distància pesa molt més un atribut que té un rang molt gran (ex. [0, 10000]) que no pas un atribut amb un rang petit
(ex. [0,1]) tot i que potser porta molta més informació sobre la classe aquest darrer. Per evitar aquesta
diferència de rangs generalment és normalitzen les dades: a cada valor d'un atribut és resta la mitja i es divideix per
la variança quedant el rang [-1, 1]. Aquesta normalizació funciona bé quan els atributs seguixen una distribució normal i
per tant, els seus valors no queden distorcionats
* Linear discriminant analysis (LDA): és una tècnica matemàtica que fa dues coses: 1. usa un canvi de base de manera que les 
classes quedin el més separades possible entre elles i 2. redueix la dimensionalitat de les dades (el nombre d'atributs)
al nombre de classes existents -1. Sempre abans d'aplicar el LDA s'han de normalitzar les dades. L'aplicació d'aquesta 
tècnica té dues conseqüències: 1. Les classes queden ben 
separades en termes de distància. 2. Com a conseqüència del canvi de base que es fa, els atributs resultants són independents
(o tendeixen a ser-ho).  
Aquesta tècnica funciona bé quan les classes es poden separar amb fronteres lineals. 

### Què teniu?
Al fitxer *classifyWin.py* trobareu

* Les dades carregades i separats el que són les dades pròpiament dites (**wine_data**) amb els seus atributs i la 
classe a on pertany cada dada (**wine_class**)

* Els preprocessadors que hem explicat abans ja preparats: el que normaliza **sc**, 
i el que fa el LDA **lda** 

* Els tres classificadors també preparats: decision tree classifier (**dtc**), Gaussian Naive Bayes (**GNB**) i K-Nearest
Neighbor (**knn**). Fixeu-vos que aquest darrer algorisme té el paràmetre *k* 

* La funció *k_folder_cross_classification(classifier, data, target_class, transform = None, folds=10, normalize=sc, lda = lda)*
que fa un K-fold cross validation i retorna els resultats d'haver fet les classificacions (per defecte 10). Teniu exemples
de com usar aquesta funció amb el decision tree i els tres preprocessats de les dades.
Si us fixeu en el codi de la funció quan fa el preprocessat de les dades primer calcula la normalització o el LDA usant
només les dades de training (i les preprocessa transformant-les) i després aplica exactament la mateixa transforamció
a les dades de testing. Dit d'uan altra forma, no usa les dades del testing per calcular la mitja ni la variança en la 
normalització ni les matrius de covariança en el LDA

* La funció *average_metrics(scores)* que calcula la mitja dels resultats d'haver aplicat 10 cops (o el que digui el
paràmetre fold de l'anterior funció) la classificació. Els resultats produïts són: precision, recall i f1-score

* Exemples de classificació amb el decision tree i els tres preprocessats de les dades

### Que heu de fer?
* Repetir l'exemple de classificació amb el decision tree però usant els altres dos mètodes de classificació. En tots els
casos has d'aplicar els tres preprocessats

* En el cas del KNN buscar quina és la millor k (recordeu que la k representa el nombre de veïns que s'usen per determinar
la classe i que generalment s'usen números imparells per evitar empats)

* Fer un informe a l'apartat següent responent les preguntes
    * Quina k funcina millor pel KNN? posa els resultats obtinguts amb els diferents k
    * Posa els resultats obtinguts amb els altes 2 algorismes
    * Els tres algorismes milloren quan es normalitzen les dades? 
        * Quin millora més? explica per què creus que passa
        * Quin empitxora? explica per què creus que passa
        * Quin es queda exactament igual? explica per què creus que passa
    * Els tres algorismes milloren quan s'aplica el LDA? Explica a què es degut per cada algorisme
