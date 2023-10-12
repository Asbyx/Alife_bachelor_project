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
- Implémenter avec les bestagones
- Etudier l'émergence de patterns de plusieurs lattices qui se maintiennent stables
- Etudier la self reproduction.

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