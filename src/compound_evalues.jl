struct MixtureEValue{D1,D2}
    numerator::D1
    denominator::D2
end

struct MixtureUniversalEValue{D}
    numerator::D
end


function (mix::MixtureEValue)(Z::EBayesSample)
    pdf(mix.numerator, Z) / pdf(mix.denominator, Z)
end

function (mix::MixtureUniversalEValue)(Z::NormalChiSquareSample)
    numerator = pdf(mix.numerator, Z)
    mean_squares = Z.mean_squares
    denominator = likelihood(Z, (λ=0.0, σ²=mean_squares))
    numerator / denominator
end