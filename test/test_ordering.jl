using Empirikos


@test NormalSample(1.0, 2.0) < NormalSample(2.0, 1.0)
@test NormalSample(1.0, 2.0) < NormalSample(1.0, 3.0)
@test NormalSample(1.0, 2.0) <= NormalSample(1.0, 2.0)


@test BinomialSample(1, 3) < BinomialSample(2, 2)
@test BinomialSample(2, 2) < BinomialSample(2, 3)
@test BinomialSample(3, 4) > BinomialSample(2, 3)
@test BinomialSample(3, 3) <= BinomialSample(3, 3)

@test PoissonSample(1, 3) < PoissonSample(2, 2)
@test PoissonSample(2, 2) < PoissonSample(2, 3)
@test PoissonSample(3, 4) > PoissonSample(2, 3)
@test PoissonSample(3, 3.1) <= PoissonSample(3, 3.1)


Zs_comp = compound( [NormalSample(1,2), NormalSample(2,1),NormalSample(1,1), NormalSample(0.5,2)])
@test Zs_comp[1] < Zs_comp[2]
@test Zs_comp[1] == Zs_comp[3]



Zs = PoissonSample.(sample(1:100, 1000))

Zs_summary = summarize(Zs)

Zs_unique = collect(keys(Zs_summary))
@test issorted(Zs_unique)

@test collect(values(Zs_summary))[2] == sum(Zs .== Zs_unique[2])
@test collect(keys(Zs_summary.store)) == collect(keys(Zs_summary))

@test issorted([response(k) for (k, v) in Zs_summary.store])
