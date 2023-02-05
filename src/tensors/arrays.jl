### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 8d6d06f2-4fd2-11ed-2fa1-451f13b4a345
md"""
## Arrays 
Julia arrays are a mutable type collection stored as a multi-dimensional grid.
Thus, they can be used to store N Dimensional Data,  N >= 100. 
`DenseArray` stores data in a contigous, column major format.\

Arrays are homogenous(type specific), which is either passed to or inferred by the constructor.
However, at a performance cost, they can be made heterogeneous using the type `Any`.
"""

# ╔═╡ 26534b74-0006-4369-89c2-65a4fa8f19a1
md"""
#### Constructors
1. Uninitalized
2. Initialized with Single Value
3. Initialized with Known Values (Literals)
4. Initialized with a Range of Values
5. Initialized with Random Values
"""

# ╔═╡ 96b194db-b947-41f5-b3a7-b0584cce2f1a
md"""
###### Uninitalized
Contents are not modified after allocation, by using the UndefInitializer().
This saves time and computation.\
`Array{Type}(undef, (dims))`
"""

# ╔═╡ 2294af6e-8003-436b-b790-78dcb2e8350e
Array{Float32}(undef, (2, 3))

# ╔═╡ 0a65e38e-0eff-40a4-bd56-9c3fc1494937
# Column Vector (1D Array)
Vector{Float32}(undef, 5)

# ╔═╡ 3265d0c6-2b83-45e1-8478-5ca2bd509836
# Row Vector (2D Array)
Array{Float32}(undef, (1, 5))

# ╔═╡ 82ffea97-3ab7-4b72-95f9-621a62ce56a2
md"""
###### Initialized with a single value
"""

# ╔═╡ c60b3198-f63c-4a22-bab2-2ecd5599a6cc
fill(π,3)

# ╔═╡ 396f3664-5fa2-490e-b34b-827d44cf0ea4
begin
    a = Array{Float32}(undef, 2, 3)
    fill!(a, ℯ)
end

# ╔═╡ 791a8f69-32d7-4830-946b-8e0cd00bf458
begin
    A = Vector{Any}(undef, 4)
    A[1] = zeros(Int32, 3)
    A[2] = ones(Int32, 3)
    A[3] = trues(3)
    A[4] = falses(3)
    A
end

# ╔═╡ d3fe5f93-f31e-4a33-ab60-466514aa2a9f
md"""
###### Initialized with known values
For array literals, type can either be provided and the elements converted, or a type can be inferred.
Concatenation can also be carried out using this method, in 2 ways.
1. Using `whitespace` for `hcat` and using `;` for `vcat` to concatenate the **contents** of the elements
2. Using 
"""

# ╔═╡ 935450aa-6c78-4a61-ae2d-6854feffd260
# Row Vector
Float32[[1 1 2] [3 5 8]]

# ╔═╡ e09c27b1-1841-46a5-84e1-2b9e4898d4eb
[[2 7]; [1 8]; [2 8]]

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─8d6d06f2-4fd2-11ed-2fa1-451f13b4a345
# ╟─26534b74-0006-4369-89c2-65a4fa8f19a1
# ╟─96b194db-b947-41f5-b3a7-b0584cce2f1a
# ╠═2294af6e-8003-436b-b790-78dcb2e8350e
# ╠═0a65e38e-0eff-40a4-bd56-9c3fc1494937
# ╠═3265d0c6-2b83-45e1-8478-5ca2bd509836
# ╟─82ffea97-3ab7-4b72-95f9-621a62ce56a2
# ╠═c60b3198-f63c-4a22-bab2-2ecd5599a6cc
# ╠═396f3664-5fa2-490e-b34b-827d44cf0ea4
# ╠═791a8f69-32d7-4830-946b-8e0cd00bf458
# ╟─d3fe5f93-f31e-4a33-ab60-466514aa2a9f
# ╠═935450aa-6c78-4a61-ae2d-6854feffd260
# ╠═e09c27b1-1841-46a5-84e1-2b9e4898d4eb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
