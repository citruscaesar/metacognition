##TODO: Use Multiple Dispatch for Char and String Types
function rotate(rot, text)
    cipher = Dict(Char(i) => Char((i+rot-Int('a'))%26 + Int('a')) for i in Int('a'):Int('z'))
    encoded = Char[] 
    for char in text
        encoded_char = char
        isuppercase(char) && (encoded_char = uppercase(cipher[lowercase(char)]))
        islowercase(char) && (encoded_char = cipher[char])
        append!(encoded, encoded_char)
    end
    return String(encoded)
end

macro ROT13_str(text)
    rotate(13, text)
end