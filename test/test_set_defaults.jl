using Empirikos
using Test

discr = DiscretePriorClass(Empirikos.DataBasedDefault())

Zs = StandardNormalSample.([1.0;2.0])
discr_instantiated = Empirikos.set_defaults(discr, Zs)
@test extrema(discr_instantiated.support) == (1.0 - 1e-4, 2.0 + 1e-4)
@test length(discr_instantiated.support) == 300

discr_instantiated_hinted = Empirikos.set_defaults(discr, Zs; hints=:prior_grid_length=>991)
@test length(discr_instantiated_hinted.support) == 991
@test extrema(discr_instantiated_hinted.support) == extrema(discr_instantiated.support)


tmp_grid = 0:0.1:10
discr_grid = DiscretePriorClass(tmp_grid)
@test Empirikos.set_defaults(discr_grid, Zs) == discr_grid


npmle = NPMLE(discr, nothing)

npmle_instantiated = Empirikos.set_defaults(npmle, Zs)
@test npmle_instantiated.convexclass == discr_instantiated

npmle_typefield = NPMLE(discr, Float64)
npmle_typefield_instantiated = Empirikos.set_defaults(npmle_typefield, Zs)
@test npmle_instantiated.convexclass == npmle_typefield_instantiated.convexclass

npmle_grid = NPMLE(discr_grid, nothing)
@test Empirikos.set_defaults(npmle_grid, Zs) == npmle_grid
