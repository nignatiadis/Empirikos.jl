export CressieSeheult

"""
    CressieSeheult

A household survey involved the participants in completing answers on question
forms which were then collected and put into batches for coding.
A quality control programme was implemented to check on coding accuracy for one question.
The table shows the numbers of errors after sampling 42 coded questionnaires
from each of 91 batches

This dataset is from the following reference:

    > Cressie, Noel, and Allan Seheult. "Empirical Bayes estimation in sampling inspection."
    Biometrika 72, no. 2 (1985): 451-458.
"""
module CressieSeheult

using CSV

const DATA = joinpath(@__DIR__, "cressie_seheult_1985.csv")

function load_table()
    CSV.File(DATA)
end

end
