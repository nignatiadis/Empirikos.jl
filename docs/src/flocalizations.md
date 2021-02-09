# F-Localizations

```@docs
Empirikos.FLocalization
Empirikos.FittedFLocalization
```
## DKW-F-Localization
```@docs
DvoretzkyKieferWolfowitz
```  
## $\chi^2$-F-Localization
```@docs
ChiSquaredFLocalization
```
## Gauss-F-Localization

```@docs
Empirikos.InfinityNormDensityBand
```
This F-Localization currently only works for homoskedastic Normal samples with common noise variance $\sigma^2$. By default the above uses the following kernel, with bandwidth $h = \sigma/\sqrt{\log(n)}$, where $n$ is the sample size:
```@docs
Empirikos.FlatTopKernel
Empirikos.SincKernel
```