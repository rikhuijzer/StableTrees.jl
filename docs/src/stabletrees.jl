### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 7c10c275-54d8-4f1a-947f-7861199cdf21
# ╠═╡ show_logs = false
# hideall
begin
	PKGDIR = dirname(dirname(@__DIR__))
	PROJECT_DIR = @__DIR__
	using Pkg: Pkg
	Pkg.activate(PROJECT_DIR)
	Pkg.develop(; path=PKGDIR)
end

# ╔═╡ 148bdc38-19e8-4dfc-80d5-ffeaee28b804
using StableRNGs

# ╔═╡ 5366a8f7-1465-4f0a-b9c8-096818104c24
using CategoricalArrays: categorical

# ╔═╡ 758d8562-f88d-47bd-82aa-22caeda9c208
using StableTrees

# ╔═╡ f833dab6-31d4-4353-a68b-ef0501d606d4
begin
	using CairoMakie
	using CSV: CSV
	using DataDeps: DataDeps, DataDep, @datadep_str
	using DataFrames: DataFrame
	using MLJDecisionTreeInterface: DecisionTree, DecisionTreeClassifier
	using MLJ: CV, MLJ, Not, auc, fit!, evaluate, machine
	using PlutoUI: TableOfContents # hide
	using StableRNGs: StableRNG
end

# ╔═╡ e9028115-d098-4c61-a82f-d4553fe654f8
# hideall
TableOfContents()

# ╔═╡ b1c17349-fd80-43f1-bbc2-53fdb539d1c0
md"""
This package implements the **S**table and **I**nterpretable **RU**le **S**ets (SIRUS) for binary classification.
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
This dataset contains observations from a study with patients who had breast cancer.
The `survival` column contains a `0` if a patient has died within 5 years and `1` if the patient has survived for at least 5 years.
The aim is to predict survival based on the `age`, the `year` in which the operation was conducted and the number of detected auxillary `nodes`.

Via [`MLJ.jl`](https://github.com/alan-turing-institute/MLJ.jl), we can fit multiple decision trees on this dataset:
"""

# ╔═╡ e5a45b1a-d761-4279-834b-216df2a1dbb5
md"""
This has fitted various trees to various subsets of the dataset.
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
This shows that although the features in the tree are the same, the values are not.
For larger datasets, you will even see that the tree can look completely different for small data-pertubations.
This is called stability.
Or in this case, a decision tree is considered to be unstable.
This unstability is problematic in situations where real-world decisions are based on the outcome of the model.
Imagine using this model for the selecting which students are allowed to enter some university.
If the model is updated every year with the data from the last year, then the selection criteria would vary wildly per year.
https://christophm.github.io/interpretable-ml-book/
This unstability also causes accuracy to fluctuate wildly.
Intuitively, this makes sense: if the model changes wildly for small data changes, then model accuracy also changes wildly.
This is why random forests were introduced.
Basically, random forests fit a large number of trees and average their predictions to come to a more accurate prediction.
The individual trees are obtained by restricting the observations and the features that the trees are allowed to use.
For the restriction on the observations, the trees are only allowed to see `partial_sampling * n` observations.
In practise, this is often `0.7`.
The restriction on the features is defined in such a way that it guarantees that not every tree will take the same split at the root of the tree.
This makes the trees less correlated (James et al., [2021](https://doi.org/10.1007/978-1-0716-1418-1; Section 8.2.2)) and, hence, very accurate.

Unfortunately, these random forests are hard to interpret.
To interpret the model, individuals would need to read hundreds to thousands of trees containing multiple levels.
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
To see how this works, let's look at the `nodes` feature on its own:
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
To solve this, Bénard et al. decided to limit the splitpoints to a pre-defined set of points.
For each feature, they find `q` empirical quantiles where `q` is typically 10.
Let's overlay these quantiles on top of the `nodes` feature:
"""

# ╔═╡ a816caed-659c-4b07-b9b2-9a820d844416
md"""
The reason that these cutpoints lay much to the left is that there are many datapoints at the left.
For most people, few auxillary `nodes` were detected.

Next, let's see where the cutpoints are when we take the same random subset as above:
"""

# ╔═╡ 52b61a65-a2d0-4ef9-b3e3-e0eb825ca501
md"""
As can be seen, many cutpoints are at the same location as before.
Furthermore, compared to the unrestricted range, the chance that two different trees who see a different random subset of the data will select the same cutpoint has increased dramatically.
"""

# ╔═╡ aa560aad-9de4-4e7f-92ce-316f88439d57
md"""
This package implements the **S**table and **I**nterpretable **RU**le **S**ets (SIRUS) for binary classification.
Regression and multiclass-classification are also technically possible but not yet implemented.
The focus is on binary classification because I need that for a paper.

The SIRUS algorithm was presented by Bénard et al. in 2020 and 2021.
In short, SIRUS combines the predictive accuracy of random forests with the explainability of decision trees while remaining stable.
Decision trees are easily interpretable but are unstable, meaning that small changes in the dataset can change the model drastically.
Random forests have solved this by fitting multiple trees.
However, interpretability of random forests is limited even with tools such as Shapley values.
For example, it is not possible to reconstruct the model given only the Shapley values.
This package solves these problems by finding a number of decision rules; typically 10.

Note that, compared to random forests, the accuracy of these rules are lower than the accuracy of the pure tree.
However, in practise, it is very likely that the accuracy of the rules is higher because the model can be verified by interpreting the rules.

## Algorithm

Decision tree-based algorithms are known to be unstable.
In other words, the trees can change drastically for small changes in the data.
This is caused by the process in which the trees choose their splits.
To choose the splits, a greedy recursive binary splitting algorithm is used (James et al., [2014](https://doi.org/10.1007/978-1-0716-1418-1)).
For example, consider the following two-dimensional case:
"""

# ╔═╡ e7861f63-aa29-419d-a458-275c8ca9bcfb
n = 10

# ╔═╡ 679abca3-9f22-4a43-a4b5-77dbea63bf08
# X = rand(StableRNG(1), n, 2);

# ╔═╡ 73666382-d78e-4096-a97b-7ed90b88d694
# y = categorical(rand(StableRNG(1), 0:1, n));

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

# ╔═╡ 93a7dd3b-7810-4021-bf6e-ae9c04acea46
_rng(seed::Int=1) = StableRNG(seed);

# ╔═╡ ed913163-b720-4f3a-978f-a844f448c923
if !haskey(ENV, "REGISTERED_HABERMAN")
    name = "Haberman"
    message = "Slightly modified copy of Haberman's Survival Data Set"
    remote_path = "https://github.com/rikhuijzer/haberman-survival-dataset/releases/download/v1.0.0/haberman.csv"
    checksum = "a7e9aeb249e11ac17c2b8ea4fdafd5c9392219d27cb819ffaeb8a869eb727a0f"
    DataDeps.register(DataDep(name, message, remote_path, checksum))
    ENV["REGISTERED_HABERMAN"] = "true"
end;

# ╔═╡ be324728-1b60-4584-b8ea-c4fe9e3466af
function _io2text(f::Function)
	io = IOBuffer()
	f(io)
	s = String(take!(io))
	return Base.Text(s)
end;

# ╔═╡ 7ad3cf67-2acd-44c6-aa91-7d5ae809dfbc
function _evaluate(model, X, y; nfolds=5)
    resampling = CV(; nfolds, shuffle=true, rng=_rng())
    acceleration = MLJ.CPUThreads()
    evaluate(model, X, y; acceleration, verbosity=0, resampling, measure=auc)
end;

# ╔═╡ 2cb02c2a-6890-40d8-9323-c3f836a03617
function _haberman()
    dir = datadep"Haberman"
    path = joinpath(dir, "haberman.csv")
    df = CSV.read(path, DataFrame)
    df[!, :survival] = categorical(df.survival)
    # Need Floats for the LGBMClassifier.
    for col in [:age, :year, :nodes]
        df[!, col] = float.(df[:, col])
    end
    return df
end;

# ╔═╡ 961aa273-d97b-497f-a79a-06bf89dc34b0
haberman = _haberman()

# ╔═╡ 6e16f844-9365-43af-9ea7-2984808f1fd5
X = haberman[:, Not(:survival)];

# ╔═╡ b6957225-1889-49fb-93e2-f022ca7c3b23
y = haberman.survival;

# ╔═╡ 9e313f2c-08d9-424f-9ea4-4a4641371360
tree_evaluations = let
	model = DecisionTreeClassifier(; max_depth=2, rng=_rng())
	_evaluate(model, X, y)
end;

# ╔═╡ 2dcd43e6-41b9-412b-b5bc-550a89376497
# hideall
let
	fig = Figure()
	ax = Axis(fig[1, 1]; xlabel="X[:, 1]", ylabel="X[:, 2]")
	scatter!(ax, X[:, 1], X[:, 2], markersize=14, marker=markers(y))
	fig
end

# ╔═╡ 39fd9deb-2a27-4c28-ae06-2a36c4c54427
let
	tree = tree_evaluations.fitted_params_per_fold[1].tree
	_io2text() do io
		DecisionTree.print_tree(io, tree; feature_names=names(haberman))
	end
end

# ╔═╡ 368b6fc1-1cf1-47b5-a746-62c5786dc143
let
	tree = tree_evaluations.fitted_params_per_fold[2].tree
	_io2text() do io
		DecisionTree.print_tree(io, tree; feature_names=names(haberman))
	end
end

# ╔═╡ 172d3263-2e39-483c-9d82-8c22059e63c3
nodes = sort(haberman.nodes);

# ╔═╡ cf1816e5-4e8d-4e60-812f-bd6ae7011d6c
# hideall
ln = length(nodes);

# ╔═╡ de90efc9-2171-4406-93a1-9a213ab32259
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	scatter!(ax, nodes, fill(1, ln))
	hideydecorations!(ax)
	fig
end

# ╔═╡ 2c1adef4-822e-4dc0-946b-dc574e50b305
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	scatter!(ax, nodes, fill(1, ln))
	vlines!(ax, [nodes[303]]; color=:red)
	hideydecorations!(ax)
	fig
end

# ╔═╡ bfcb5e17-8937-4448-b090-2782818c6b6c
# hideall
subset = collect(ST._rand_subset(_rng(), nodes, round(Int, 0.7 * ln)));

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

# ╔═╡ 8b57cda0-7249-440d-90b2-ff4ca27e6d6c
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	cutpoints = ST._cutpoints(subset, 10)
	scatter!(ax, subset, fill(1, ls))
	vlines!(ax, cutpoints; color=:red)
	hideydecorations!(ax)
	fig
end

# ╔═╡ 3d68d35b-2192-4640-895d-51ce8e29a368
# hideall
let
	fig = Figure(; resolution=(800, 100))
	ax = Axis(fig[1, 1])
	cutpoints = ST._cutpoints(nodes, 10)
	scatter!(ax, nodes, fill(1, ln))
	vlines!(ax, cutpoints; color=:red)
	hideydecorations!(ax)
	fig
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
# ╠═3d68d35b-2192-4640-895d-51ce8e29a368
# ╠═a816caed-659c-4b07-b9b2-9a820d844416
# ╠═8b57cda0-7249-440d-90b2-ff4ca27e6d6c
# ╠═52b61a65-a2d0-4ef9-b3e3-e0eb825ca501
# ╠═aa560aad-9de4-4e7f-92ce-316f88439d57
# ╠═148bdc38-19e8-4dfc-80d5-ffeaee28b804
# ╠═e7861f63-aa29-419d-a458-275c8ca9bcfb
# ╠═679abca3-9f22-4a43-a4b5-77dbea63bf08
# ╠═5366a8f7-1465-4f0a-b9c8-096818104c24
# ╠═73666382-d78e-4096-a97b-7ed90b88d694
# ╠═83700ef9-f833-49d0-9ee9-76eb56f643e9
# ╠═2dcd43e6-41b9-412b-b5bc-550a89376497
# ╠═758d8562-f88d-47bd-82aa-22caeda9c208
# ╠═0ca8bb9a-aac1-41a7-b43d-314a4029c205
# ╠═0e0252e7-87a8-49e4-9a48-5612e0ded41b
# ╠═e1890517-7a44-4814-999d-6af27e2a136a
# ╠═f833dab6-31d4-4353-a68b-ef0501d606d4
# ╠═93a7dd3b-7810-4021-bf6e-ae9c04acea46
# ╠═ed913163-b720-4f3a-978f-a844f448c923
# ╠═be324728-1b60-4584-b8ea-c4fe9e3466af
# ╠═7ad3cf67-2acd-44c6-aa91-7d5ae809dfbc
# ╠═2cb02c2a-6890-40d8-9323-c3f836a03617
