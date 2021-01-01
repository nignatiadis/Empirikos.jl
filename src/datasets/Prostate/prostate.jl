export Prostate

module Prostate

using CSV
using ..Empirikos: MultinomialSummary, StandardNormalSample

const URL = "https://web.stanford.edu/~hastie/CASI_files/DATA/prostz.txt"

function load_table()
    CSV.File(download(URL), header=false)
end

function ebayes_samples()
    tbl = load_table()
    Zs = StandardNormalSample.(tbl.Column1)
end

end
