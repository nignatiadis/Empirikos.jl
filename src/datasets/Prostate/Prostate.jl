export Prostate

"""
    Prostate

The dataset is from the following reference:

> Dinesh Singh, Phillip G. Febbo, Kenneth Ross, Donald G. Jackson, Judith Manola,
 Christine Ladd, Pablo Tamayo, Andrew A. Renshaw, Anthony V. D’Amico, Jerome P. Richie,
 Eric S. Lander, Massimo Loda, Philip W. Kantoff, Todd R. Golub, and William R. Sellers.
 Gene expression correlates of clinical prostate cancer behavior.
 Cancer cell, 1(2): 203–209, 2002.

See the following monograph for further illustrations
of empirical Bayes methods on this dataset:

> Bradley Efron. Large-scale inference: Empirical Bayes methods for estimation, testing,
  and prediction. Cambridge University Press, 2012
"""
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
