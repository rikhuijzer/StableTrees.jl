### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 7c10c275-54d8-4f1a-947f-7861199cdf21
# ╠═╡ show_logs = false
# hideall
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

	PKGDIR = dirname(dirname(@__DIR__))
	PROJECT_DIR = @__DIR__
	using Pkg: Pkg
	Pkg.activate(PROJECT_DIR)
	Pkg.develop(; path=PKGDIR)
end

# ╔═╡ f833dab6-31d4-4353-a68b-ef0501d606d4
begin
	using CairoMakie
	using CategoricalArrays: categorical
	using CSV: CSV
	using MLDatasets: BostonHousing
	using DataFrames: DataFrame, Not, dropmissing!, select!
	using LightGBM.MLJInterface: LGBMClassifier
	using MLJDecisionTreeInterface: DecisionTree, DecisionTreeClassifier
	using MLJ: CV, MLJ, Not, PerformanceEvaluation, auc, fit!, evaluate, machine
	using PlutoUI: TableOfContents # hide
	using StableRNGs: StableRNG
	using StableTrees: StableTrees, StableForestClassifier, StableRulesClassifier
	using Statistics: mean
end

# ╔═╡ e9028115-d098-4c61-a82f-d4553fe654f8
# hideall
TableOfContents()

# ╔═╡ b1c17349-fd80-43f1-bbc2-53fdb539d1c0
md"""
This package implements the **S**table and **I**nterpretable **RU**le **S**ets (SIRUS) for binary classification.
(Regression and multiclass-classification will be implemented later.)
SIRUS is based on random forests.
However, compared to random forests, the results are much easier to explain since the forests are converted to a set of decison rules.
This page will provide an overview of the algorithm and describe not only how it can be used but also how it works.
To do this, let's start by briefly describing random forests.
"""

# ╔═╡ 348d1235-87f2-4e8f-8f42-be89fef5bf87
md"""
## Random forests

Random forests are known to produce accurate predictions especially in settings where the number of features `p` is close to or higher than the number of observations `n` (Biau & Scornet, [2016](https://doi.org/10.1007/s11749-016-0481-7)).
Let's start by explaining the building blocks of random forests: decision trees.
As an example, we take Haberman's Survival Data Set (see the Appendix below for more details):
"""

# ╔═╡ 4c8dd68d-b193-4846-8d93-ab33512c3fa2
md"""
This dataset contains information on [housing in Boston, MA](http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html).
The `MEDV` column contains a `0` if the house price is below the mean of all the known values of occupied homes.
Specifically, the column contains a `0` if the house price is below 22.5 thousand dollars and a `1` if the house price is below 22.5 thousand dollars.
The mean house price is so low because of inflation; the dataset is from 1978.
Anyway, as a last check, the dataset is reasonably well balanced:
"""

# ╔═╡ f75aa57f-6e84-4f7e-88e4-11a00cb9ad2b
md"""
Via [`MLJ.jl`](https://github.com/alan-turing-institute/MLJ.jl), we can fit multiple decision trees on this dataset:
"""

# ╔═╡ e5a45b1a-d761-4279-834b-216df2a1dbb5
md"""
This has fitted various trees to various subsets of the dataset via cross-validation.
Here, I've set `max_depth=2` to simplify the fitted trees which makes the tree more easily explainable.
Also, for our small dataset, this forces the model to remain simple so it likely reduces overfitting.
Let's look at the first tree:
"""

# ╔═╡ d38f8814-c7b8-4911-9c63-d99b646b4486
md"""
What this shows is that the first tree decided that the `nodes` feature is the most helpful in deciding who will survive for 5 more years.
Next, if the `nodes` feature is below 2.5, then `age` will be selected on.
If `age < 79.5`, then the model will predict the second class and if `age ≥ 79.5` it will predict the first class.
Similarly for `age < 43.5`.
Now, let's see what happens for a slight change in the data.
In other words, let's see how the fitted model for the second split looks:
"""

# ╔═╡ 5318414e-5c87-4be0-bcd0-b6efd4eee5b9
md"""
This shows that the features and the values for the splitpoints are not the same for both trees.
This is called stability.
Or in this case, a decision tree is considered to be unstable.
This unstability is problematic in situations where real-world decisions are based on the outcome of the model.
Imagine using this model for the selecting which students are allowed to enter some university.
If the model is updated every year with the data from the last year, then the selection criteria would vary wildly per year.
This unstability also causes accuracy to fluctuate wildly.
Intuitively, this makes sense: if the model changes wildly for small data changes, then model accuracy also changes wildly.
This intuitively also implies that the model is more likely to overfit.
This is why random forests were introduced.
Basically, random forests fit a large number of trees and average their predictions to come to a more accurate prediction.
The individual trees are obtained by restricting the observations and the features that the trees are allowed to use.
For the restriction on the observations, the trees are only allowed to see `partial_sampling * n` observations.
In practise, `partial_sampling` is often 0.7.
The restriction on the features is defined in such a way that it guarantees that not every tree will take the same split at the root of the tree.
This makes the trees less correlated (James et al., [2021](https://doi.org/10.1007/978-1-0716-1418-1; Section 8.2.2)) and, hence, very accurate.

Unfortunately, these random forests are hard to interpret.
To interpret the model, individuals would need to interpret hundreds to thousands of trees containing multiple levels.
Alternatively, methods have been created to visualize these uninterpretable models (for example, see Molnar ([2022](https://christophm.github.io/interpretable-ml-book/)); Chapters 6, 7 and 8).
The most promising one of these methods are Shapley values and SHAP.
These methods show which features have the highest influence on the prediction.
See my blogpost on [Random forests and Shapley values](https://huijzer.xyz/posts/shapley/) for more information.
Knowing which features have the highest influence is nice, but they do not state exactly what feature is used and at what cutoff.
Again, this is not good enough for selecting students into universities.
For example, what if the government decides to ask for details about the selection?
The only answer that you can give is that some features are used for selection more than others and that they are on average used in a certain direction.
If the government asks for biases in the model, then these are impossible to report.
In practise, the decision is still a black-box.
SIRUS solves this by extracting easily interpretable rules from the random forests.
"""

# ╔═╡ d816683b-2f7d-45a7-bd40-42f554a48b1b
md"""
## Rule-based models

Rule-based models promise much greater interpretability than random forests.
Instead of returning a large number of trees, rule-based models return a set of rules.
Each rule can be interpreted on its own and the final model aggregates these rules by summing the prediction of each rules.
For example, one rule can be:

> if `nodes < 4.5` then chance of survival is 0.6 and if `nodes ≥ 4.5` then chance of survival is 0.4.

Note that these rules can be extracted quite easily from the decision trees.
For splits on the second level of the tree, the rule could look like:

> if `nodes < 4.5` and `age < 38.5` then chance of survival is 0.8 and otherwise the chance of survival is 0.4.

When applying this extracting of rules to a random forest, there will be thousands of rules.
Next, via some heuristic, the most important rules can be localized and these rules then result in the final model.
See, for example, RuleFit (Friedman & Popescu, [2008](https://www.jstor.org/stable/30245114)).
The problem with this approach is that they are fitted on the unstable decision trees that were shown above.
As an example, on time the tree splits on `age < 43.5` and another time on `age < 44.5`.
"""

# ╔═╡ 4b67c47a-ee98-495e-bb1b-41db83c11cd4
md"""
## Tree stabilization

In the papers which introduce SIRUS, Bénard et al. ([2021a](https://doi.org/10.1214/20-EJS1792), [2021b](https://proceedings.mlr.press/v130/benard21a.html)) proof that their algorithm is stable and that the other algorithms are not.
They achieve their stability by restricting the location at which the splitpoints can be chosen.
To see how this works, let's look at the `NOX` feature on its own.
The NOX feature gives the number of nitric oxides in parts per 10 million:
"""

# ╔═╡ 0d121fa3-fbfa-44e5-904b-64a1622ec91b
md"""
The default random forest algorithm is allowed to choose any location inside this feature to split on.
To avoid having to figure out locations by itself, the algorithm will choose on of the datapoints as a split location.
So, for example, the following split indicated by the red vertical line would be a valid choice:
"""

# ╔═╡ 896e00dc-2ce9-4a9f-acc1-519aec21dd83
md"""
But what happens if we take a random subset of the data?
Say, we take the following subset of length `0.7 * length(nodes)`:
"""

# ╔═╡ ee12350a-627b-4a11-99cb-38c496977d18
md"""
Now, the algorithm would choose a different location and, hence, introduce unstability.
To solve this, Bénard et al. decided to limit the splitpoints that the algorithm can use to split to data to a pre-defined set of points.
For each feature, they find `q` empirical quantiles where `q` is typically 10.
Let's overlay these quantiles on top of the `NOX` feature:
"""

# ╔═╡ a816caed-659c-4b07-b9b2-9a820d844416
md"""
The reason that these cutpoints lay much to the left is that there are many datapoints at the left.
For most people, few auxillary `nodes` were detected.

Next, let's see where the cutpoints are when we take the same random subset as above:
"""

# ╔═╡ 01b08d44-4b9b-42e2-bb20-f34cb9b407f3
md"""
As can be seen, many cutpoints are at the same location as before.
Furthermore, compared to the unrestricted range, the chance that two different trees who see a different random subset of the data will select the same cutpoint has increased dramatically.
"""

# ╔═╡ 6cb0ded0-8f49-498a-8fe9-7ce3ea10d945
md"""
The benefit of this is that it is now quite easy to extract the most important rules.
It works by extracting the rules and simplifying them a bit.
Next, the rules can be ordered by frequency of occurence to get the most important rules.
Let's see how accurate this model is.
"""

# ╔═╡ 7e1d46b4-5f93-478d-9105-a5b0db1eaf08
md"""
## Benchmark

For the benchmark, lets compare the following models:

- Decision tree (`DecisionTreeClassifier`)
- Stabilized random forest (`StableForestClassifier`)
- SIRUS (`StableRulesClassifier`)
- LightGBM (`LGBMClassifier`)

The latter is a state-of-the-art gradient boosting model created by Microsoft.
"""

# ╔═╡ 1d08ca81-a18a-4a74-992c-14243d2ea7dc
function _score(e::PerformanceEvaluation)
	return round(only(e.measurement); digits=2)
end;

# ╔═╡ 4dcd564a-5b2f-4eae-87d6-c2973b828282
_filter_rng(hyper::NamedTuple) = Base.structdiff(hyper, (; rng=:foo));

# ╔═╡ 7a9a0242-a7ba-4508-82fd-a48084525afe
_pretty_name(modeltype) = last(split(string(modeltype), '.'));

# ╔═╡ 6a539bb4-f51f-4efa-af48-c43318ed2502
_hyper2str(hyper::NamedTuple) = hyper == (;) ? "(;)" : string(hyper)::String;

# ╔═╡ cece10be-736e-4ee1-8c57-89beb0608a92
function _evaluate(modeltype, hyperparameters, X, y)
    model = modeltype(; hyperparameters...)
	e = evaluate(model, X, y)
	row = (;
	    Model=_pretty_name(modeltype),
	    Hyperparameters=_hyper2str(_filter_rng(hyperparameters)),
	    AUC=_score(e),
	    se=round(only(MLJ.MLJBase._standard_errors(e)); digits=2)
	)
end;

# ╔═╡ 83700ef9-f833-49d0-9ee9-76eb56f643e9
# hideall
markers(Y) = [y == 1 ? :circle : :cross for y in Y]

# ╔═╡ 0ca8bb9a-aac1-41a7-b43d-314a4029c205
ST = StableTrees;

# ╔═╡ 0e0252e7-87a8-49e4-9a48-5612e0ded41b
md"""
## Acknowledgements

Thanks to Clément Bénard, Gérard Biau, Sébastian da Veiga and Erwan Scornet for creating the SIRUS algorithm and documenting it extensively.
Special thanks to Clément Bénard for answering my questions regarding the implementation.
"""

# ╔═╡ e1890517-7a44-4814-999d-6af27e2a136a
md"""
## Appendix
"""

# ╔═╡ ede038b3-d92e-4208-b8ab-984f3ca1810e
function _plot_cutpoints(data::AbstractVector)
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	cutpoints = Float64.(unique(ST._cutpoints(data, 10)))
	scatter!(ax, data, fill(1, length(data)))
	vlines!(ax, cutpoints; color=:black, linestyle=:dash)
	textlocs = [(c, 1.1) for c in cutpoints]
	for cutpoint in cutpoints
		annotation = string(round(cutpoint; digits=2))::String
		text!(ax, cutpoint + 0.003, 1.08; text=annotation, textsize=11)
	end
	ylims!(ax, 0.9, 1.2)
	hideydecorations!(ax)
	return fig
end;

# ╔═╡ 93a7dd3b-7810-4021-bf6e-ae9c04acea46
_rng(seed::Int=1) = StableRNG(seed);

# ╔═╡ be324728-1b60-4584-b8ea-c4fe9e3466af
function _io2text(f::Function)
	io = IOBuffer()
	f(io)
	s = String(take!(io))
	return Base.Text(s)
end;

# ╔═╡ 7ad3cf67-2acd-44c6-aa91-7d5ae809dfbc
function _evaluate(model, X, y; nfolds=10)
    resampling = CV(; nfolds, shuffle=true, rng=_rng())
    acceleration = MLJ.CPUThreads()
    evaluate(model, X, y; acceleration, verbosity=0, resampling, measure=auc)
end;

# ╔═╡ 2cb02c2a-6890-40d8-9323-c3f836a03617
function _boston()
    data = BostonHousing()
    df = hcat(data.features, data.targets)
    dropmissing!(df)
    for col in names(df)
        df[!, col] = float.(df[:, col])
    end
    # Median value of owner-occupied homes in 1000's of dollars.
    target = :MEDV
    m = mean(df[:, target]) # 22.5 thousand dollars.
    df[!, target] = categorical([value < m ? 0 : 1 for value in df[:, target]])
	select!(df, target, :)
	select!(df, Not(:B))
	return df
end;

# ╔═╡ 961aa273-d97b-497f-a79a-06bf89dc34b0
boston = _boston()

# ╔═╡ 6e16f844-9365-43af-9ea7-2984808f1fd5
X = boston[:, Not(:MEDV)];

# ╔═╡ b6957225-1889-49fb-93e2-f022ca7c3b23
y = boston.MEDV;

# ╔═╡ 48110693-1aee-4af7-878d-0ae9a545657d
length(filter(==(0), y))

# ╔═╡ 4dc13f14-41cf-4589-a057-ef69aee783f8
length(filter(==(1), y))

# ╔═╡ 9e313f2c-08d9-424f-9ea4-4a4641371360
tree_evaluations = let
	model = DecisionTreeClassifier(; max_depth=2, rng=_rng())
	_evaluate(model, X, y)
end;

# ╔═╡ ab103b4e-24eb-4575-8c04-ae3fd9ec1673
e1 = let
	model = DecisionTreeClassifier
	hyperparameters = (; max_depth=2, rng=_rng(3))
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 6ea43d21-1cc0-4bca-8683-dce67f592949
# ╠═╡ show_logs = false
e2 = let
	model = StableForestClassifier
	hyperparameters = (; rng=_rng())
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 88a708a7-87e8-4f97-b199-70d25ba91894
# ╠═╡ show_logs = false
e3 = let
	model = StableRulesClassifier
	hyperparameters = (; max_rules=10, rng=_rng())
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 5d875f9d-a0aa-47b0-8a75-75bb280fa1ba
# ╠═╡ show_logs = false
e4 = let
	model = StableRulesClassifier
	hyperparameters = (; max_rules=25, rng=_rng())
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 6ca70265-ede3-4efd-86fa-e6940a45e84f
# ╠═╡ show_logs = false
e5 = let
	model = LGBMClassifier
	hyperparameters = (; max_depth=2)
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 263ea81f-5fd6-4414-a571-defb1cabab4b
# ╠═╡ show_logs = false
e6 = let
	model = LGBMClassifier
	hyperparameters = (; )
	_evaluate(model, hyperparameters, X, y)
end;

# ╔═╡ 622beb62-51ac-4b44-9409-550e5f422fe4
let
	df = DataFrame([e1, e2, e3, e4, e5, e6])
end

# ╔═╡ 39fd9deb-2a27-4c28-ae06-2a36c4c54427
let
	tree = tree_evaluations.fitted_params_per_fold[1].tree
	_io2text() do io
		DecisionTree.print_tree(io, tree; feature_names=names(boston))
	end
end

# ╔═╡ 368b6fc1-1cf1-47b5-a746-62c5786dc143
let
	tree = tree_evaluations.fitted_params_per_fold[2].tree
	_io2text() do io
		DecisionTree.print_tree(io, tree; feature_names=names(boston))
	end
end

# ╔═╡ 172d3263-2e39-483c-9d82-8c22059e63c3
nox = sort(boston.NOX);

# ╔═╡ cf1816e5-4e8d-4e60-812f-bd6ae7011d6c
# hideall
ln = length(nox);

# ╔═╡ de90efc9-2171-4406-93a1-9a213ab32259
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	scatter!(ax, nox, fill(1, ln))
	hideydecorations!(ax)
	fig
end

# ╔═╡ 2c1adef4-822e-4dc0-946b-dc574e50b305
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	scatter!(ax, nox, fill(1, ln))
	vlines!(ax, [nox[300]]; color=:red)
	hideydecorations!(ax)
	fig
end

# ╔═╡ bfcb5e17-8937-4448-b090-2782818c6b6c
# hideall
subset = collect(ST._rand_subset(_rng(3), nox, round(Int, 0.7 * ln)));

# ╔═╡ dff9eb71-a853-4186-8245-a64206379b6f
# hideall
ls = length(subset);

# ╔═╡ 25ad7a18-f989-40f7-8ef1-4ca506446478
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	scatter!(ax, subset, fill(1, ls))
	hideydecorations!(ax)
	fig
end

# ╔═╡ 1471aac2-140f-4b2f-a3e6-15bca257f9f6
_plot_cutpoints(subset)

# ╔═╡ 4935d8f5-32e1-429c-a8c1-84c242eff4bf
_plot_cutpoints(nox)

# ╔═╡ ef3605ec-93dc-4b9f-b4f1-3014b881c349
let
	c1 = unique(ST._cutpoints(nox, 10))
	c2 = (ST._cutpoints(subset, 10))
	count(c1 .== c2)
end

# ╔═╡ Cell order:
# ╠═7c10c275-54d8-4f1a-947f-7861199cdf21
# ╠═e9028115-d098-4c61-a82f-d4553fe654f8
# ╠═b1c17349-fd80-43f1-bbc2-53fdb539d1c0
# ╠═348d1235-87f2-4e8f-8f42-be89fef5bf87
# ╠═961aa273-d97b-497f-a79a-06bf89dc34b0
# ╠═6e16f844-9365-43af-9ea7-2984808f1fd5
# ╠═b6957225-1889-49fb-93e2-f022ca7c3b23
# ╠═4c8dd68d-b193-4846-8d93-ab33512c3fa2
# ╠═48110693-1aee-4af7-878d-0ae9a545657d
# ╠═4dc13f14-41cf-4589-a057-ef69aee783f8
# ╠═f75aa57f-6e84-4f7e-88e4-11a00cb9ad2b
# ╠═9e313f2c-08d9-424f-9ea4-4a4641371360
# ╠═e5a45b1a-d761-4279-834b-216df2a1dbb5
# ╠═39fd9deb-2a27-4c28-ae06-2a36c4c54427
# ╠═d38f8814-c7b8-4911-9c63-d99b646b4486
# ╠═368b6fc1-1cf1-47b5-a746-62c5786dc143
# ╠═5318414e-5c87-4be0-bcd0-b6efd4eee5b9
# ╠═d816683b-2f7d-45a7-bd40-42f554a48b1b
# ╠═4b67c47a-ee98-495e-bb1b-41db83c11cd4
# ╠═172d3263-2e39-483c-9d82-8c22059e63c3
# ╠═cf1816e5-4e8d-4e60-812f-bd6ae7011d6c
# ╠═de90efc9-2171-4406-93a1-9a213ab32259
# ╠═0d121fa3-fbfa-44e5-904b-64a1622ec91b
# ╠═2c1adef4-822e-4dc0-946b-dc574e50b305
# ╠═896e00dc-2ce9-4a9f-acc1-519aec21dd83
# ╠═bfcb5e17-8937-4448-b090-2782818c6b6c
# ╠═dff9eb71-a853-4186-8245-a64206379b6f
# ╠═25ad7a18-f989-40f7-8ef1-4ca506446478
# ╠═ee12350a-627b-4a11-99cb-38c496977d18
# ╠═4935d8f5-32e1-429c-a8c1-84c242eff4bf
# ╠═1471aac2-140f-4b2f-a3e6-15bca257f9f6
# ╠═a816caed-659c-4b07-b9b2-9a820d844416
# ╠═01b08d44-4b9b-42e2-bb20-f34cb9b407f3
# ╠═ef3605ec-93dc-4b9f-b4f1-3014b881c349
# ╠═6cb0ded0-8f49-498a-8fe9-7ce3ea10d945
# ╠═7e1d46b4-5f93-478d-9105-a5b0db1eaf08
# ╠═1d08ca81-a18a-4a74-992c-14243d2ea7dc
# ╠═4dcd564a-5b2f-4eae-87d6-c2973b828282
# ╠═7a9a0242-a7ba-4508-82fd-a48084525afe
# ╠═6a539bb4-f51f-4efa-af48-c43318ed2502
# ╠═cece10be-736e-4ee1-8c57-89beb0608a92
# ╠═ab103b4e-24eb-4575-8c04-ae3fd9ec1673
# ╠═6ea43d21-1cc0-4bca-8683-dce67f592949
# ╠═88a708a7-87e8-4f97-b199-70d25ba91894
# ╠═5d875f9d-a0aa-47b0-8a75-75bb280fa1ba
# ╠═6ca70265-ede3-4efd-86fa-e6940a45e84f
# ╠═263ea81f-5fd6-4414-a571-defb1cabab4b
# ╠═622beb62-51ac-4b44-9409-550e5f422fe4
# ╠═83700ef9-f833-49d0-9ee9-76eb56f643e9
# ╠═0ca8bb9a-aac1-41a7-b43d-314a4029c205
# ╠═0e0252e7-87a8-49e4-9a48-5612e0ded41b
# ╠═e1890517-7a44-4814-999d-6af27e2a136a
# ╠═ede038b3-d92e-4208-b8ab-984f3ca1810e
# ╠═f833dab6-31d4-4353-a68b-ef0501d606d4
# ╠═93a7dd3b-7810-4021-bf6e-ae9c04acea46
# ╠═be324728-1b60-4584-b8ea-c4fe9e3466af
# ╠═7ad3cf67-2acd-44c6-aa91-7d5ae809dfbc
# ╠═2cb02c2a-6890-40d8-9323-c3f836a03617
