export PsychologyReproducibility
# cleaned_tbl <- data.frame(study=d$Study.Title..O., pvalue = d$T_pval_USE..R., absolute_zscore = abs(qnorm(d$T_pval_USE..R./2)))
# write.csv(cleaned_tbl, "reproducibility_study.csv", row.names=FALSE)
module PsychologyReproducibility

using CSV
using ..Empirikos: StandardNormalSample

const DATA = joinpath(@__DIR__, "reproducibility_study.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    StandardNormalSample.(tbl.absolute_zscore)
end


end
