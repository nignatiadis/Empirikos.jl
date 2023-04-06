
function StatsBase.fit(mle::ParametricMLE{<:Gamma}, Zs)
    as_map = as(Vector, asℝ₊, 2)
    init_params = [0.0; 0.0]
    params_to_gamma(p) = Gamma(transform(as_map,p)...)
    ℓ(p) = -loglikelihood(Zs, params_to_gamma(p))
    opt = Optim.optimize(ℓ, init_params, mle.solver)
    params_to_gamma(opt.minimizer)
end
