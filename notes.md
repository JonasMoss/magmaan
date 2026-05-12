## Structure
* [ ] **partable** should define the model *modulo* ideas such as normality, polychorics, GLS, or even t-distributed residuals or any other crazyness. 
  * It should ONLY contain the information required for such models.
    * Estimates, se, start -> someplace else.
    * Separate `partable` and `start`. 
  * Suggested names: LatentStructure.
* [ ] Lets go through all the columns in the table and (dis)confirm they are needed.
  * Why `plabel`? Is it always identical to `id`? If so, there's no need to store it.
  * Why `label`? Does it do anything? (ah, this is to keep user labels...)
  * Why `block`? Is it identical to group?
  * How about `user`? This also looks completely useless.
* We might separate the table in two: One for the actual structure (independent of names), one for names and labels. Maybe better separation?

## Optimization
* [ ] Use unit variance for identification, not setting arbitrary params to 1.

## Estimation methods
* GLS family: Figure out a setup for this; do we supply a weighting matrix or make separate functions?

##
* [ ] Change name from latva -> magmaan (joke name lol)
  

## Misc
* pt   <- latva_lavaanify(m, n_groups = 2L, group_var = "school") <- group_var doesn't do anything to the `pt`, but it should. we must store the group name since its part of the verbal model. (ideally we'd keep the names of the groups too, as names of the variables.)