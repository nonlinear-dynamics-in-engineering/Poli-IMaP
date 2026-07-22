mutable struct Polynomial{D}
    # object to hold the coefficients of a polynomial of a certain order (max degree)
    # coefficients are stored as a flat vector; index mapping is in ExpoTableFlat.expo_to_idx
    order::Int
    ddim::Int
    tensor::Vector{D}
end

# zero polynomial constructor
Polynomial{D}() where D = Polynomial{D}(0, 1, D[D(0)])

# constant polynomial constructor
Polynomial{D}(ct::D) where D = Polynomial{D}(0, 1, D[ct])

# constructor of standard polynomial — nterms = C(ddim+order, ddim)
Polynomial{D}(order::Int, ddim::Int) where D =
    Polynomial{D}(order, ddim, zeros(D, binomial(ddim + order, ddim)))
