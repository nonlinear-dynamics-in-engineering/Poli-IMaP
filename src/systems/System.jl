Base.@kwdef mutable struct System{E}
    ddim::Int
    rdim::Int
    F::Function
    B::Union{Nothing, Matrix{E}, Function} = nothing
    Bpattern::Union{Nothing, Vector{Tuple{Int,Int}}} = nothing
    maxorder::Int
    D::DataType
    initialized::Bool = false
    gF::Graph{Polynomial{E}} = Graph{Polynomial{E}}()
    gFn::Graph{E} = Graph{E}()
    gB::Graph{Polynomial{E}} = Graph{Polynomial{E}}()
    gBn::Graph{E} = Graph{E}()
    jacobian::Any = []
    eigvals::Any = []
    r_eigvecs::Any = []
    l_eigvecs::Any = []
    B₀::Union{Nothing, Matrix{E}} = nothing
    flatB₀::Any = []
    invB₀::Any = []
    B_is_const::Bool = false
    B_sparse::Bool = false
    B_rowmap::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[]
    W::PolynomialArray{E} = PolynomialArray{E}()
    f::PolynomialArray{E} = PolynomialArray{E}()
    DW::PolynomialArray{E} = PolynomialArray{E}()

    DWk::PolynomialArray{E} = PolynomialArray{E}()
    DWf::PolynomialArray{E} = PolynomialArray{E}()
    BDWf::Union{Nothing, PolynomialArray{E}} = nothing
    W_vec::Vector{Polynomial{E}} = Polynomial{E}[]
end
