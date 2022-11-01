### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ c8a42d83-db76-4a2f-bf87-198b50e340ea
md"""
### Tuples
"""

# ╔═╡ 1fdec622-58e2-11ed-37f8-876b89cdf1be
md"""
Tuples, similiar to arrays, are also ordered type collections, but in contrast to arrays they are **immutable**, 
and are charaterized comma seperated values within a parentheses ```(a, b, c, ...)```.\
"""

# ╔═╡ 457f383e-13e7-4b26-8ef2-f17513e2bacb
begin
    t = (1, 8, 11)
    typeof(t)
end

# ╔═╡ 92247a74-44c8-4922-bc0e-b92d186bdcf1
md"""
In Julia, they serve as an abstraction of the arguments of a function, thus the important properties of function arugments,
i.e. the order and types are important for tuples.
"""

# ╔═╡ f7f02034-71ce-4f80-841b-20b91bcaac07
typeof((13, "something", 2.71))

# ╔═╡ b65c42ea-fb20-4f9f-a87e-11031b75b45c
md"""
Unpacking tuples is similar to python
"""

# ╔═╡ 3f814464-025f-4806-a371-2d612aed0355
a, b, c = t

# ╔═╡ 52e301d5-023c-4d79-ad61-6904a6f7c9ea
md"""
### Named Tuples
Named Tuples with a single entry 
```(;key = value)``` to distinguish from a normal variable
"""

# ╔═╡ e7a4c99c-5ffc-4670-be7a-eb7e3e4e7c5a
begin
    nt = (
        x = 1,
        y = 2.5,
        z = "text"
    )
    x, y = nt
    x, y
end

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
# ╟─c8a42d83-db76-4a2f-bf87-198b50e340ea
# ╟─1fdec622-58e2-11ed-37f8-876b89cdf1be
# ╠═457f383e-13e7-4b26-8ef2-f17513e2bacb
# ╟─92247a74-44c8-4922-bc0e-b92d186bdcf1
# ╠═f7f02034-71ce-4f80-841b-20b91bcaac07
# ╟─b65c42ea-fb20-4f9f-a87e-11031b75b45c
# ╠═3f814464-025f-4806-a371-2d612aed0355
# ╟─52e301d5-023c-4d79-ad61-6904a6f7c9ea
# ╠═e7a4c99c-5ffc-4670-be7a-eb7e3e4e7c5a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
