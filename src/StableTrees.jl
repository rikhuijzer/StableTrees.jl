module StableTrees

import AbstractTrees: children, nodevalue

using Random: AbstractRNG
using Tables: Tables

const Float = Float32

include("forest.jl")

include("interface.jl")
StableForestClassifier = Interface.StableForestClassifier
export StableForestClassifier

end # module
