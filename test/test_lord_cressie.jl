# We compare the chi^2 F-localization intervals against the intervals reported
# by Lord and Cressie.


lord_cressie = LordCressie.load_table()
Zs = LordCressie.ebayes_samples()

idx_test = [1, 13, 20]
z_cressie = lord_cressie.x[idx_test]
lower_test = lord_cressie.Lower1[idx_test]
upper_test = lord_cressie.Upper1[idx_test]

gcal = DiscretePriorClass(range(0.0,stop=1.0,length=300));

postmean_targets = Empirikos.PosteriorMean.(BinomialSample.(z_cressie,20));
chisq_nbhood = Empirikos.ChiSquaredNeighborhood(0.05)

nbhood_method_chisq = NeighborhoodWorstCase(neighborhood = chisq_nbhood,
                                       convexclass= gcal, solver=Hypatia.Optimizer)

chisq_cis = confint.(nbhood_method_chisq, postmean_targets, Zs)

lower_chisq_ci = getproperty.(chisq_cis, :lower)
upper_chisq_ci = getproperty.(chisq_cis, :upper)

@test lower_chisq_ci ≈ lower_test atol=1e-3
@test upper_chisq_ci ≈ upper_test atol=0.005
