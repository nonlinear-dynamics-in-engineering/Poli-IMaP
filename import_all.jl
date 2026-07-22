using Revise, LinearAlgebra, SparseArrayKit, Symbolics, Printf, ProgressMeter
using Base: reduce, product

function import_all()
    includet("./src/graphs/graphs.jl");
    includet("./src/polynomials/Polynomial.jl")
    includet("./src/polynomials/PolynomialArray.jl")
    includet("./src/polynomials/utils.jl")

    includet("./src/graphs/num/nodedata.jl");
    includet("./src/graphs/num/autodiff.jl");
    includet("./src/graphs/num/build.jl");

    includet("./src/graphs/poly/nodedata.jl");
    includet("./src/graphs/poly/autodiff.jl");
    includet("./src/graphs/poly/build.jl");

    includet("./src/systems/System.jl")
    includet("./src/systems/utils.jl")
    includet("./banner.jl")
    includet("./src/symbolic/spectral_interpolation.jl")

    includet("./src/parametrizations/ParSettings.jl")
    includet("./src/parametrizations/utils.jl");

    includet("./src/parametrizations/solve.jl");
    includet("./src/parametrizations/Bmatrix.jl");
    includet("./src/parametrizations/residues.jl");

    includet("./src/parametrizations/aut.jl");
    includet("./src/parametrizations/nonaut.jl");
end

nothing

import_all()

show_banner()
