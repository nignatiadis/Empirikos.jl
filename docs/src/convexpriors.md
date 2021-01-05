# Convex prior classes

The starting point for many empirical Bayes tasks, such as inference or estimation, is to posit
that the true prior $G$ lies in a convex class of priors $\mathcal{G}$. Such classes of priors 
are represented in this package through the abstract type,

```@docs
Empirikos.ConvexPriorClass
```

Currently, the following choices for $\mathcal{G}$ are available:

```@docs
DiscretePriorClass
MixturePriorClass
GaussianScaleMixtureClass
```

