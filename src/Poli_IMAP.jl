#=

Poli-IMaP: [Poli]'s [I]nvariant [Ma]nifold [P]arametrizer

Written by: Paolo F. Ferrari, 2026
Contact: paoloff@usp.br

Standalone package module. Load with "using Poli_IMAP"

=#
module Poli_IMAP

using LinearAlgebra
using SparseArrayKit
using Symbolics
using Printf
using ProgressMeter
using Base: reduce, product

include("graphs/graphs.jl")
include("polynomials/Polynomial.jl")
include("polynomials/PolynomialArray.jl")
include("polynomials/utils.jl")

include("graphs/num/nodedata.jl")
include("graphs/num/autodiff.jl")
include("graphs/num/build.jl")

include("graphs/poly/nodedata.jl")
include("graphs/poly/autodiff.jl")
include("graphs/poly/build.jl")

include("systems/System.jl")
include("systems/utils.jl")
include("../banner.jl")
include("symbolic/spectral_interpolation.jl")

include("parametrizations/ParSettings.jl")
include("parametrizations/utils.jl")
include("parametrizations/solve.jl")
include("parametrizations/Bmatrix.jl")
include("parametrizations/residues.jl")
include("parametrizations/aut.jl")
include("parametrizations/nonaut.jl")

# public 
export System,
       initialize_system!, linearize_system!,
       set_parametrization_settings,
       parametrize_autonomous!, parametrize_nonautonomous!,
       ParSettings, Polynomial, PolynomialArray,
       homog_exponents, homog_components,
       show_banner

function __init__()
    show_banner()
end

end # module Poli_IMAP
