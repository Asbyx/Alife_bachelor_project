# Terminologie:

|Paper|Ici|Signification|
|-|-|-|
|Lattice|Cell|L'endroit qui contient les différents channels|  
|Particule|Particule|Une particule qui se déplace et qui subit les lois d'interaction|
|Channel|Channel|Recueil à particules, il y en a 2 types: rest et communication|


# Meeting Vass 04 octobre
Ai réussi à implémenter GOL et LGCA classic avec torch, présentation du résultat, c'est joli.  

### *Explication de mes idées pour le BIO-LGCA*:  
Il faut que je choisisse un scope, i.e est ce que je veux que mon organisme vivant soit une particule, les particules dans le rest channel, ou plusieurs hexagones. À noter qu'au niveau particule et particules dans un rest channel, la reproduction est triviale à partir de règles faites main, il n'y aura rien d'émergent.  
→ Vass trouve que le plus intéressant est de faire un organisme multicellulaire, donc une cellule par hexagone qui communiquent entre elles pour avoir des behaviors intéressants.

### *Input de Vass:*  
Un comportement intéressant à implémenter serait que chaque cell puisse savoir à quel niveau de depth dans l'organisme elle est, i.e les cellules au bord sont à un depth de 0, et les cellules au centre, entourée d'autres cellules, sont à un niveau plus haut.  
Ce behavior peut être intéressant car il ouvre la voie vers une reprodction par ARN, i.e chaque cellule catch les éléments pour sa reproduction, puis quand tout le monde a réussi à catch elles le communiquent avec les autres, release le matériel puis bim: dublication.  

Ça serait bien que je réussisse à implémenter le BIO-LGCA avec torch, pour des mega simulations, mais si c'est trop compliqué et que ça me fait perdre du temps ça en faut pas la peine du tout, on passe en cpu tranquille. Le but est quand même de pousser le modèle BIO-LGCA et en tirer des choses intéressantes.


# Meeting Vass 12 octobre
Ait montré mon implémentation du BIO-LGCA, Vass dit que la partie aléatoire on s'en fout un peu, si on en veut on en met dans l'interaction function.   
Par contre concernant le fait que normalement l'interaction function dépend des lattices autour, il dit que c'est mieux si on continue sur notre lancée actuelle car 1) c'est plus proche de la réalité des cellules qui doivent communiquer entre elles 2) ça sera plus simple à implémenter sur la GPU   
Il dit aussi que l'hexagone est important à terme parce que 1) y'a pas de coins, ce qui est mieux 2) les cellules en diagonales devraient être des voisines mais ne le sont pas dans une implémentation carrée.

On a aussi discuté qu'en essence, la fonction d'interaction c'est simplement une fonction qui va de rest + communication channels à rest + communication, donc très facile de faire du data parallelizing avec.

### *Feuille de route*
- Une fois la depth implémentée, on essaye de partir d'un rest state "oeuf" qui doit arriver à une figure définie. On essaye ça d'abord avec un carré. Le but est de poser une particule qui contient la longueur d'un demi-côté puis qu'elle évolue en un carré complet. Should be quite easy  
- Une fois qu'on a l'oeuf, on passe à la poule, i.e un truc qui bouge. Donc le but est de montrer qu'on peut faire se déplacer des patterns. *Ça serait bien que j'arrive à faire en sorte que tout se déplace d'un coup et pas layer par layer. Aussi, faire une technique qui permet de faire se déplacer n'importe quel pattern.*

### *Autres pistes*
- Faire un modèle qui grow un pattern qu'on définit en avance. Optimalement même, le pattern est encodé en tant qu'ADN dans le lattice
- Implémenter avec les bestagones
- Etudier l'émergence de patterns de plusieurs lattices qui se maintiennent stables
- Etudier la self reproduction.


# Meeting Vass 19 octobre
Je continue d'implémenter les particules qui bougent seules et s'aggrègent, dans le but de faire une reproduction par ARN. C'est à dire qu'on veut implémenter des paires de lattices qui catchent d'autres lattices bougeant pour se reproduire.  

Ça donnerait: un lattice A collisionne avec un G. Ils forment alors une paire qui va essayer de se reproduire. Ils catchent respectivement un T et un C puis les détachent. La nouvelle paire va alors avancer de TBD steps, avant de s'arrêter et essayer aussi de se reproduire.  
Ça permet de faire une simulation particulesque et assez proche de la réalité.

Sinon, on fait qqch de plus simple type: on pose un oeuf, il grow une créature, cette créature envoie des signaux comme un oursin, quand 2 signaux se rencontrent ils forment une nouvelle créature qui combine l'ADN qu'il y avait dans les signaux.  
Ça permet de faire une simulation qui est bcp plus du type "cellular automata" mais avec une évolution darwinienne vraiment facile à faire.


# Meeting Vass 09 novembre
Pas grand chose de nouveau je continue sur la reproduction des paires d'ADN.  
Pour l'autre direction, aka les bulles qui envoient des oeufs, on verra plus tard là on focus sur ADN.

Les pistes pour continuer actuellement: 
- implémenter la séparation du child *(fait)*
- faire la différenciation des particules (A, T, G, C) *(fait)*
- mort des paires au bout d'un moment *(fait)*
- chaînes plus longues ?
- faire un meilleur code parce que là c'est catastrophique:
  - Séparer les interactions dans des sous-fonctions, pour que ce soit lisible et + facile de modifier un comportement spécifique
  - Mettre plus de channels dans les comm channels pour pouvoir faire passer + d'infos
  - Tout passer en bit signals pour implémenter l'optimisation par dictionnaire
  - Refaire en Java ? en soi j'ai juste besoin de créer une window qui me permet d'afficher des pixels, ou tout enregistré et mettre dans une vidéo


# Meeting Vass 23 novembre
On continue selon les pistes vues au dessus.
Pour les chaînes plus longues, on a brainstormé un peu, la solution la + simple est de faire qqch d'elastique, c'est à dire que quand une chaine veut bouger, simplement elle fait avancer 1 lattice, puis les autres de la chaîne se rendent compte que leur voisin manque du coup ils avancent.  
Et pour détecter quand bouger, le plus simple serait de trouver un moyen pour les lattices d'une chaîne de connaître la longueur de leur chaîne. Puis quand quand la chaîne fille se rend compte qu'elle a la longueur de la chaîne mère, elle se décroche. Ou alors une fois que tous les lattices parents acknowledgent qu'ils ont grab, on relâche un lattice child au pif, puis l'elasticité fera le reste (les bouts de chaines préviennent quand ils ont bougé permettant au leader de la chaine d'avancer quand tout le monde est là).   
Le plus gros problème c'est les obstacles. Mais en y réfléchissant, c'est pas trop un problème car les free vont juste dégager au bout d'un moment, permettant à l'elasticité de continuer. Sauf si on rencontre une autre chaîne fixée, mais on pourrait du coup dissoudre la chaîne qui voyage, où une chaîne qui bouge, là c'est plus relou. Une solution ce serait de faire des réservations et de ne bouger que lorsque tout le monde est prêt.
Reflexion to be continued.


# Meeting Vass 30 novembre
La différenciation a été effectuée ! Mais du coup on a très très peu de paires qui arrivent à se reproduire (calculer la proba est faisable et intéressant ?).
Je démarre le coding de la mort des paires, décidé à 5xlongueur_coté_monde


# Meeting Vass 7 decembre
Les reproductibles pairs sont finies ! La mort a été implémentée, ça permet aussi d'avoir plus de reproduction overall. Par contre, vu que la complexité du modèle a drastiquement augmenté, c'est plus difficile de se rendre compte de l'apparition de comportements complexes.  
Prochaine étape: faire un Game Of Life ! Parce que ça permettra de bien montrer que le modèle est capable de faire des simulations physiques, des CA et de la simulation de particules.  
Après ça, rédaction du rapport à fond ! 

### Séparation du child:
La grosse difficulté est de faire bouger une paire de lattice en gérant les collisions. Dans la vie réelle d'ailleurs ça n'existe pas trop donc on peut laisser tomber.  
Ce qui serait bien c'est que cette reproduction influe l'environnement. Le cas trivial et celui proposé par Vass est que ça créé une autre paire qui essaye à son tour de se reproduire.
Après c'est useless parce qu'il n'y a pas de mort des paires pour le moment, donc aucun sélection.

Options:
- Faire des particules oeufs qui se déplacent de N unité puis éclosent -> pb de la place encore une fois mais gérable avec des réservations
- Faire des résas comme d'hab, sauf que les paires ont la prio. Donc protocole: les 2 lattices envoient une resa puis:
  - Si c'est libre: ils informent leur paire que c'est libre
  - Si les 2 sont libres (chaque lattice regarde lui même + signal de la paire) -> move
  - Si 1 n'est pas libre -> wait

Choix: faire les résas comme d'hab, avec du coup un nouveau state pour les paris qui migrent:  
Nouveau signal: reservation_pair, qui la priorité sur les réservations de lattice.  
Chaque paire envoie à l'autre si la case devant elle est dispo, et quand les deux sont ok -> ça bouge avec une seed  
Cas particuliers:
- Si un free rencontre une resa_pair qui va dans la direction opposée à lui, il flip
- Si la travelling pair rencontre une pair en train de grab, à travers un signal grabbing, **la travelling pair est dissoute** (i.e les lattices deviennent free et partent dans des directions opposées)




# Random ideas  
**Notion d'aléatoire dans le paper:** pour le moment j'implémente un deterministic bio lgca. L'aléatoire décrite dans le paper est contenue dans la fonction d'interaction. En fait le passage d'un state à un autre est simplement une variable aléatoire *dépendante des cells voisines*. Un aspect que j'ai pour le moment complètement omis.

**DNA and phenotype:** DNA of a cell could be encoded in one or several particules. To do that, simply use a different interaction function when a particular cell is resting in a rest channel. We could also implement that, by default the interaction function is the id function, with a random chance of 2 particules colliding to lose their momentum (2 particles in opposite channels go in rest channel). Ideally, the dna is all rest channels (it is the combination of particles that makes the DNA).

**Hexagonal implementation:** https://www.redblobgames.com/grids/hexagons/

**Origine de la vie simulée:** https://www.youtube.com/watch?v=Jdaz5e_a5xk => pas mal de choses à tirer, je pense je dois modifier mon BIO-LGCA pour que ce soit + intéressant et plus proche de la réalité  
- Conservation de la matière, pour faire une chimie plus plausible. Peut être relou à implémenter et à calculer, mais empêcher l'apparition spontanée de particules sera déjà un bon début.  
Il est aussi à noter que dans la vie, on utilise la matière disponible et on la réarrange pour que ça marche, on ne crée rien. Cette notion d'utilisation de ce qui est à disposition me semble assez essentielle.
- point intéressant: pour avoir de l'évolution darwinienne il faut passer par un étape où des patterns auto reproducteurs peuvent émerger d'une situation chaotique aléatoire, et donc doivent être assez simples puis se complexifier grâce à l'évolution darwinienne  
Il est aussi intéressant de noter que le replicateur n'a pas besoin de popper par hasard, on peut avoir un simple pattern qui débouche sur l'auto repro., un peu à la manière de la fourmie de jsplus qui, qui suit un pattern chaotique avant de rentrer dans une boucle.  

Une fois qu'on sait se faire se déplacer des patterns, essayer de faire se déplacer des patterns qui évoluent en se déplaçant.

**Pour synchroniser des lattices** *(genre pour drop des grandes chaines ARN)* on peut simplement leur donner une clock interne et certaines actions ne peuvent être faites que si la clock est à 0  
*Problème de la clock:* Si tu as un signal qui requiert que qqch du type (`lattice à coté` && `moi`), même si tu les synchronises avec un clock, il peut y avoir une situation où "moi" ne communique pas avec le lattice d'à côté, et donc celui d'à coté ne reçoit pas l'info qu'il doit bouger

**L'évolution requiert la mort, pour le moment y'a pas de mort dans BIO-LGCA**

**Difficultés de ce modèle:**  
- On veut jouer avec le mouvement, mais le mouvement de lattices est compliqué et doit être harcodé.  
En fait cela me conforte dans l'idée que l'automaton de Clément est un peu l'automaton ultime, car les particules sont libres de bouger comme elles veulent, ce qui est bcp plus réaliste quand même, mais le fait de run une sorte de cellular automata entre les cellules proches permet de simuler les envois de protéines ou jsp quoi entre les cellules. A creuser.

**Optimisations random:** 
On peut coder les interactions avec des if-else, puis un compilateur run tous les cas possibles pour faire un wati dictionnaire. Comme ça quand on run la simulation on peut simplement aller chercher dans le dico.  
Encore plus simple: Si tous les channels ne sont que des 0 et des 1, alors un lattice est simplement une suite binaire, ce qui peut être mappé sur un nombre. Pour faire le dico, on peut donc simplement rentrer tous les nombres possibles, voir vers quel nombre il est mappé et bim, on a notre dico qui tournera en O(1)

**Pistes de recherches**
J'aimerais bien faire de la recherche sur un algo simples qui trouve des solutions, un peu comme genetic algo, mais dans un CA ou particles simulation

**Comportement non prévu du biolgca**: 2 organismes peuvent former une clock intéressantes (ils s'envoient des travelling pairs lul). Et je pense qu'on peut montrer que c'est Turing complete :) parce que des portes sont faisables

# Rendu:
L'oral doit intéresser clément, il est intéressé par un petit talk (arn = intéressé de fou).
Date: le plus tard c'est le mieux parce que ça donne le temps de faire des choses.
Il veut pas que je le fasse pour la note, il veut que je le fasse parce que c'est un talk intéressant.
