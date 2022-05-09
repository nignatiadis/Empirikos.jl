using KernelDensity
using Random
using QuadGK
Random.seed!(1)

Zs = StandardNormalSample.(randn(2000) .* sqrt(2))

kernel1 = Empirikos.FlatTopKernel(1.0)
kernel2 = Empirikos.FlatTopKernel(0.2)


@test quadgk(z->pdf(kernel1, z), -50.0, 50.0)[1] ≈ 1.0 atol = 0.005
@test quadgk(z->pdf(kernel2, z), -50.0, 50.0)[1] ≈ 1.0 atol = 0.005

kde1 = kde(response.(Zs), kernel2; boundary = (-12.0,12.0), npoints=2048*4)

idxs = [2000; 2048*2; 6000]
for idx in idxs
    x1 = kde1.x[idx]
    kde1_f = kde1.density[idx]
    kde1_manual = mean( pdf.(kernel2, response.(Zs) .- x1) )
    @test kde1_f ≈ kde1_manual atol = 0.001
end


# check the full fit using InfinityNormDensityBand
# at α = 0.5 and α = 0.2
Zs_half_std = NormalSample.(response.(Zs), 0.5)

floc_02 = InfinityNormDensityBand(a_min=-3.0, a_max=3.0, α =0.2 )
floc_05 = InfinityNormDensityBand(a_min=-3.0, a_max=3.0, α =0.5 )

floc_fit_02 = fit(floc_02, Zs)
floc_fit_05 = fit(floc_05, Zs)

floc_fit_half_02 = fit(floc_02, Zs_half_std)

@test std(floc_fit_02.midpoints[1]) == 1.0
@test std(floc_fit_half_02.midpoints[1]) == 0.5

# check default bandwidth is what we expect

@test floc_fit_02.method.kernel.h == 1/sqrt(log(2000))
@test floc_fit_05.method.kernel.h == floc_fit_02.method.kernel.h
@test floc_fit_half_02.method.kernel.h == 0.5/sqrt(log(2000))

# check c_infty_boot are the same in the two cases
@test floc_fit_02.boot_samples == floc_fit_05.boot_samples
@test floc_fit_05.C∞ == median(floc_fit_05.boot_samples)
@test floc_fit_02.C∞  == quantile(floc_fit_02.boot_samples, 0.8)
@test floc_fit_02.C∞ > floc_fit_05.C∞

# double check KDE fits using explicit formula as above
@test floc_fit_02.estimated_density == floc_fit_05.estimated_density
@test floc_fit_02.midpoints == floc_fit_05.midpoints

@test length(floc_fit_02.midpoints) == length(floc_fit_02.estimated_density)
rand_idxs = sample(1:length(floc_fit_02.midpoints), 20)

_kernel = floc_fit_02.method.kernel

for idx in rand_idxs
    loc = response(floc_fit_02.midpoints[idx])
    dens = floc_fit_02.estimated_density[idx]
    dens_manual = mean( pdf.(_kernel, response.(Zs) .- loc) )
    @test dens ≈ dens_manual atol = 0.001
end
