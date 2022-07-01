module StableTrees

import AbstractTrees: children, nodevalue
import Base

using CategoricalArrays: CategoricalValue, unwrap
using Random: AbstractRNG
using Tables: Tables

const Float = Float32

include("forest.jl")
include("rules.jl")
export TreePath

include("mlj.jl")
const StableForestClassifier = MLJImplementation.StableForestClassifier
export StableForestClassifier

end # module