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
Named Tuples are constructed using a tuple of ```key = value```
pairs. Ones with a single entry are written as ```(;key = value)``` 
or ```(key = value,)```to distinguish from a normal variable.
"""

# ╔═╡ b09fc60b-a058-4c68-b6c3-88c6cbf009d8
(m = 5,)

# ╔═╡ e7a4c99c-5ffc-4670-be7a-eb7e3e4e7c5a
nt = (
    x = 1,
    y = 2.5,
    z = "text"
)

# ╔═╡ 975704ff-a211-4323-b7e1-74e27064622b
md"""
There is a key difference between named tuples and dictionaries.😆
The keys in a named tuple can only be an identifier, i.e. a name or a symbol,
not a literal as in dictionaries. 
"""

# ╔═╡ c510f186-6a52-4fe8-981f-1a4775fdd401
# Accssed by key value
nt[2]

# ╔═╡ 4208947d-5fe9-4420-bf48-57d27d69c491
# Accessed by identifier
nt.y

# ╔═╡ 699d11af-611b-46bd-8dd3-0a6076d1b2dc
md"""
Named Tuples are an excellent placeholder for structs while writing prototype code.
"""

# ╔═╡ cc234b88-0c74-4515-ac18-a19285412f01
# Constructs a Named Tuple with Parameters
function person(name, age, height_in_meters)
    return (name = name, age = age, height = height_in_meters)
end

# ╔═╡ cbfdf971-3573-4cb1-ac95-d5780351fedd
# Constructs a Type with Parameters
struct Person
    name::AbstractString
    age::Int8
    height::Float16
end

# ╔═╡ 43ec35c5-e424-4906-b4b6-c439924a3cef
#boi = person("Rohit", 15, 1.6)
boi = Person("Rohit", 15, 1.6)

# ╔═╡ 168cd614-8495-4d30-ad83-a24aa6ec54f7
md"""
The next cell will work for both the struct and the named tuple.
"""

# ╔═╡ ebd9c6f3-3fb6-496d-99f6-87e3bf5eef72
#Print Details
print("Hello, I am $(boi.name). I am $(boi.age) years old and $(boi.height * 100)cm tall.")

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
# ╠═b09fc60b-a058-4c68-b6c3-88c6cbf009d8
# ╠═e7a4c99c-5ffc-4670-be7a-eb7e3e4e7c5a
# ╟─975704ff-a211-4323-b7e1-74e27064622b
# ╠═c510f186-6a52-4fe8-981f-1a4775fdd401
# ╠═4208947d-5fe9-4420-bf48-57d27d69c491
# ╠═699d11af-611b-46bd-8dd3-0a6076d1b2dc
# ╠═cc234b88-0c74-4515-ac18-a19285412f01
# ╠═cbfdf971-3573-4cb1-ac95-d5780351fedd
# ╠═43ec35c5-e424-4906-b4b6-c439924a3cef
# ╟─168cd614-8495-4d30-ad83-a24aa6ec54f7
# ╠═ebd9c6f3-3fb6-496d-99f6-87e3bf5eef72
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
