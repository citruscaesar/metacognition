"""
    ispangram(input)

Return `true` if `input` contains every alphabetic character (case insensitive).

"""
function ispangram(sentence::AbstractString)
    letters = Set{Char}(lowercase(sentence))
    alphabet = Set{Char}('a':'z')
    alphabet ⊆ letters
end

@time ispangram("The Quick Brown Fox Jumps Over The Lazy Dog")
