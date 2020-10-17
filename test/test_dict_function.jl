using Test
using DataStructures

as = rand(10)
bs = rand(10)

dict1 = Dict(as .=> bs)
dict2 = SortedDict( as .=> bs)

f1 = DictFunction(dict1)
f2 = DictFunction(dict2)

@test f1.(as) == f2.(as)
@test f1(rand()) == 0.0
@test f2(rand()) == 0.0

f3 = DictFunction(dict1, 3.0)

@test f3.(as) == f1.(as)
@test f3(rand()) == 3.0
