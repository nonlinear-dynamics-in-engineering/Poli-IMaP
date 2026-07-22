mutable struct PolynomialArray{D}
    # used to represent 1D arrays of polynomials
    # coefficients stored as a matrix: randim × nterms
    order::Int
    ddim::Int
    randim::Int
    tensors::Matrix{D}
end

# empty constructor
PolynomialArray{D}() where D = PolynomialArray{D}(0, 1, 1, reshape(D[D(0)], 1, 1))

# zero constructor
PolynomialArray{D}(zero::Number) where D = PolynomialArray{D}()

# constructor of standard array — nterms = C(ddim+order, ddim)
PolynomialArray{D}(order::Int, ddim::Int, randim::Int) where D =
    PolynomialArray{D}(order, ddim, randim,
        zeros(D, randim, binomial(ddim + order, ddim)))
