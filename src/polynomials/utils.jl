# Module-level flat index map — set once in generate_exponents!
# Used by utility functions (homog_components, update_poly!, etc.)
const _EXPO_IDX_MAP = Ref{Any}(nothing)

# SparsePartials — sparse representation of partial derivatives.
# only stores non-zero (index, value) pairs; indices are Int32, kept sorted.

struct SparsePartials{D}
    indices::Vector{Int32}
    values::Vector{D}
end
SparsePartials{D}() where D = SparsePartials{D}(Int32[], D[])

# Multiply a SparsePartials by a scalar — shares the indices vector (never mutated).
@inline function _scale(c::D, vals::Vector{D}) where D
    isempty(vals) && return vals
    out = Vector{D}(undef, length(vals))
    @inbounds for i in eachindex(vals)
        out[i] = c * vals[i]
    end
    return out
end

# Merge two scaled SparsePartials: result = a*sp1 + b*sp2.
# Pre-allocates output to max possible size to avoid push! reallocation.
function _sparse_add_scaled(a::D, sp1::SparsePartials{D}, b::D, sp2::SparsePartials{D}) where D
    n1, n2 = length(sp1.indices), length(sp2.indices)
    if n1 == 0 && n2 == 0; return SparsePartials{D}(); end
    if n1 == 0; return SparsePartials{D}(sp2.indices, _scale(b, sp2.values)); end
    if n2 == 0; return SparsePartials{D}(sp1.indices, _scale(a, sp1.values)); end
    result_idx = Vector{Int32}(undef, n1 + n2)
    result_val = Vector{D}(undef, n1 + n2)
    i1 = i2 = cnt = 1
    @inbounds while i1 <= n1 && i2 <= n2
        idx1, idx2 = sp1.indices[i1], sp2.indices[i2]
        if idx1 < idx2
            result_idx[cnt] = idx1; result_val[cnt] = a * sp1.values[i1]; i1 += 1
        elseif idx1 > idx2
            result_idx[cnt] = idx2; result_val[cnt] = b * sp2.values[i2]; i2 += 1
        else
            result_idx[cnt] = idx1; result_val[cnt] = a * sp1.values[i1] + b * sp2.values[i2]
            i1 += 1; i2 += 1
        end
        cnt += 1
    end
    @inbounds while i1 <= n1
        result_idx[cnt] = sp1.indices[i1]; result_val[cnt] = a * sp1.values[i1]; i1 += 1; cnt += 1
    end
    @inbounds while i2 <= n2
        result_idx[cnt] = sp2.indices[i2]; result_val[cnt] = b * sp2.values[i2]; i2 += 1; cnt += 1
    end
    resize!(result_idx, cnt - 1)
    resize!(result_val,  cnt - 1)
    return SparsePartials{D}(result_idx, result_val)
end

# Sorted merge with pre-allocated output (no push! reallocation).
function _sparse_add(sp1::SparsePartials{D}, sp2::SparsePartials{D}) where D
    return _sparse_add_scaled(D(1), sp1, D(1), sp2)
end

function _sparse_sub(sp1::SparsePartials{D}, sp2::SparsePartials{D}) where D
    return _sparse_add_scaled(D(1), sp1, D(-1), sp2)
end

# Flat-indexed typed containers — use Int32 indices into Vector/Matrix storage.
# Eliminates SparseArray hash lookups 
# Exponents stored as Vector{Int} 
struct ExpoTableFlat{N}
    expos::Dict{Int, Vector{Vector{Int}}}
    idxs::Dict{Int, Vector{Int32}}
    expo_to_idx::Dict{Vector{Int}, Int32}
    nterms::Int
    prod::Dict{Tuple{Int,Int}, Vector{Tuple{Int32,Int32,Int32}}}
end
Base.getindex(et::ExpoTableFlat, k::Int) = et.expos[k]
flat_idxs(et::ExpoTableFlat, k::Int) = et.idxs[k]
prod_triples(et::ExpoTableFlat, k::Int, j::Int) = et.prod[(k, j)]

struct FullExpoTableFlat{N}
    expos::Dict{Tuple{Int,Int}, Vector{Vector{Int}}}
    idxs::Dict{Tuple{Int,Int}, Vector{Int32}}
    expo_to_idx::Dict{Vector{Int}, Int32}
    nterms::Int
    prod::Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}}, Vector{Tuple{Int32,Int32,Int32}}}
end
Base.getindex(et::FullExpoTableFlat, k::Tuple{Int,Int}) = et.expos[k]
flat_idxs(et::FullExpoTableFlat, k::Tuple{Int,Int}) = et.idxs[k]
prod_triples(et::FullExpoTableFlat, k::Tuple{Int,Int}, j::Tuple{Int,Int}) = et.prod[(k, j)]

struct ConvTableFlat{N}
    conv::Dict{Int, Vector{Tuple{Int32,Int32,Int32}}}
end
Base.getindex(ct::ConvTableFlat, k::Int) = ct.conv[k]

struct FullConvTableFlat{N}
    conv::Dict{Tuple{Int,Int}, Vector{Tuple{Int32,Int32,Int32}}}
end
Base.getindex(ct::FullConvTableFlat, k::Tuple{Int,Int}) = ct.conv[k]


# homog_exponents — enumerate all n-tuples (1-indexed) whose sum equals k+n.
# returns Vector{Vector{Int}}.

function homog_exponents(n::Int, k::Int)
    result = Vector{Int}[]
    buf    = ones(Int, n)
    _fill_homog_expos!(result, buf, n, k, 1)
    return result
end

function _fill_homog_expos!(result, buf, n, remaining, pos)
    if pos == n
        buf[pos] = remaining + 1
        push!(result, copy(buf))
        return
    end
    for v in remaining:-1:0
        buf[pos] = v + 1
        _fill_homog_expos!(result, buf, n, remaining - v, pos + 1)
    end
end

function homog_exponents(N::Vector{Int}, K::Vector{Int})
    if length(N) != length(K)
        error("Both lists must have the same length")
    end
    subgroups = [homog_exponents(N[i], K[i]) for i in eachindex(N)]
    result = Vector{Int}[]
    for combo in product(subgroups...)
        push!(result, vcat(combo...))
    end
    return result
end

# returns the homogeneous polynomial of order k of a Polynomial
function homog_components(P::Polynomial{D}, hexpos::Vector) where D
    m = _EXPO_IDX_MAP[]
    result = SparseArray(zeros(D, length(hexpos)))
    @inbounds for (i, expo) in enumerate(hexpos)
        result[i] = P.tensor[m[expo]]
    end
    return result
end

function homog_components(P::Vector{Polynomial{D}}, hexpos::Vector) where D
    m = _EXPO_IDX_MAP[]
    result = SparseArray(zeros(D, length(P), length(hexpos)))
    @inbounds for (i, expo) in enumerate(hexpos), (axis, poly) in enumerate(P)
        result[axis, i] = poly.tensor[m[expo]]
    end
    return result
end

function homog_components(P::PolynomialArray{D}, hexpos::Vector) where D
    m = _EXPO_IDX_MAP[]
    result = SparseArray(zeros(D, P.randim, length(hexpos)))
    @inbounds for (i, expo) in enumerate(hexpos), axis in 1:P.randim
        result[axis, i] = P.tensors[axis, m[expo]]
    end
    return result
end


# add the coefficients inside a PolynomialArray to another PolynomialArray
function update_poly!(P::PolynomialArray, Q::PolynomialArray,
    hexpos::Vector; addToCurrent::Bool=false)
    m = _EXPO_IDX_MAP[]
    if addToCurrent
        @inbounds for expo in hexpos, axis in 1:P.randim
            ii = m[expo]
            P.tensors[axis, ii] += Q.tensors[axis, ii]
        end
    else
        @inbounds for expo in hexpos, axis in 1:P.randim
            ii = m[expo]
            P.tensors[axis, ii] = Q.tensors[axis, ii]
        end
    end
end

function update_poly!(P::PolynomialArray, V::Vector{Polynomial},
    hexpos::Vector; addToCurrent::Bool=false)
    m = _EXPO_IDX_MAP[]
    if addToCurrent
        @inbounds for expo in hexpos, axis in 1:P.randim
            ii = m[expo]
            P.tensors[axis, ii] += V[axis].tensor[ii]
        end
    else
        @inbounds for expo in hexpos, axis in 1:P.randim
            ii = m[expo]
            P.tensors[axis, ii] = V[axis].tensor[ii]
        end
    end
end

function update_poly!(P::PolynomialArray, M::SparseArray{Complex{Float64}},
    hexpos::Vector; addToCurrent::Bool=false)
    m = _EXPO_IDX_MAP[]
    if addToCurrent
        @inbounds for (expoidx, expo) in enumerate(hexpos), axis in 1:P.randim
            ii = m[expo]
            P.tensors[axis, ii] += M[axis, expoidx]
        end
    else
        @inbounds for (expoidx, expo) in enumerate(hexpos), axis in 1:P.randim
            ii = m[expo]
            P.tensors[axis, ii] = M[axis, expoidx]
        end
    end
end

function update_poly!(P::PolynomialArray, M::SparseArray{Complex{Float64}},
    hexpos::Vector, hexposidxs::Vector, axisidxs::Vector{Int};
    addToCurrent::Bool)
    m = _EXPO_IDX_MAP[]
    if addToCurrent
        @inbounds for (exponentidx, expo) in zip(hexposidxs, hexpos), axis in axisidxs
            ii = m[expo]
            P.tensors[axis, ii] += M[axis, exponentidx]
        end
    else
        @inbounds for (exponentidx, expo) in zip(hexposidxs, hexpos), axis in axisidxs
            ii = m[expo]
            P.tensors[axis, ii] = M[axis, exponentidx]
        end
    end
end

function update_poly!(P::PolynomialArray, M::SparseArray{Complex{Float64}},
    hexpos::Vector, axisidxs::Vector{Int};
    addToCurrent::Bool)
    m = _EXPO_IDX_MAP[]
    if addToCurrent
        @inbounds for (expoidx, expo) in enumerate(hexpos), axis in axisidxs
            ii = m[expo]
            P.tensors[axis, ii] += M[axis, expoidx]
        end
    else
        @inbounds for (expoidx, expo) in enumerate(hexpos), axis in axisidxs
            ii = m[expo]
            P.tensors[axis, ii] = M[axis, expoidx]
        end
    end
end


# Gradient ∇!
# Uses a pre-allocated lookup buffer to avoid allocation in the hot loop.

function ∇!(P::PolynomialArray{D}, grad::PolynomialArray{D},
    expos::ExpoTableFlat{N}, k::Int) where {D, N}
    isempty(expos[k]) && return
    buf = Vector{Int}(undef, length(first(expos[k])))
    axispairs = product(1:P.randim, 1:P.ddim)
    for (axispairidx, axispair) in enumerate(axispairs)
        rangeaxes, derivaxes = axispair
        @inbounds for (expo, ii) in zip(expos[k], flat_idxs(expos, k))
            expo[derivaxes] == 1 && continue
            coeff = expo[derivaxes] - 1
            buf .= expo
            buf[derivaxes] -= 1
            jj = expos.expo_to_idx[buf]
            grad.tensors[axispairidx, jj] += coeff * P.tensors[rangeaxes, ii]
        end
    end
end

function ∇!(P::PolynomialArray{D}, grad::PolynomialArray{D},
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where {D, N}
    isempty(expos[k]) && return
    buf = Vector{Int}(undef, length(first(expos[k])))
    axispairs = product(1:P.randim, 1:P.ddim)
    for (axispairidx, axispair) in enumerate(axispairs)
        rangeaxes, derivaxes = axispair
        @inbounds for (expo, ii) in zip(expos[k], flat_idxs(expos, k))
            expo[derivaxes] == 1 && continue
            coeff = expo[derivaxes] - 1
            buf .= expo
            buf[derivaxes] -= 1
            jj = expos.expo_to_idx[buf]
            grad.tensors[axispairidx, jj] += coeff * P.tensors[rangeaxes, ii]
        end
    end
end


# evaluate polynomial at point x.
function evaluate(P::Polynomial{D}, x::Vector{Complex{Float64}}) where D
    m = _EXPO_IDX_MAP[]
    isnothing(m) && return D(0)
    y = D(0)
    for (expo, idx) in m
        v = P.tensor[idx]
        iszero(v) && continue
        monomial = one(D)
        @inbounds for i in eachindex(x)
            monomial *= x[i]^(expo[i]-1)
        end
        y += v * monomial
    end
    return y
end

function evaluate(P::PolynomialArray, x::Vector{Complex{Float64}})
    return [evaluate(Polynomial{eltype(P.tensors)}(P.order, P.ddim, P.tensors[i, :]), x)
            for i in 1:P.randim]
end


function array_to_vec(P::PolynomialArray{D}) where D
    vec = Vector{Polynomial{D}}(undef, P.randim)
    for i in 1:P.randim
        vec[i] = Polynomial{D}(P.order, P.ddim, P.tensors[i, :])
    end
    return vec
end

function update_vec!(vec::Vector{Polynomial{D}}, P::PolynomialArray{D},
    expos::ExpoTableFlat{N}, k::Int) where {D, N}
    @inbounds for i in flat_idxs(expos, k), j in 1:P.randim
        vec[j].tensor[i] = P.tensors[j, i]
    end
end

function update_vec!(vec::Vector{Polynomial{D}}, P::PolynomialArray{D},
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where {D, N}
    @inbounds for i in flat_idxs(expos, k), j in 1:P.randim
        vec[j].tensor[i] = P.tensors[j, i]
    end
end

# dot! with dense partials vector (kept for backwards compatibility)
function dot!(vec::Vector{D}, X::PolynomialArray{D}, Y::Polynomial{D},
    k::Tuple{Int,Int}, expos::FullExpoTableFlat{N}) where {D, N}
    @inbounds for ii in flat_idxs(expos, k), i in 1:X.randim
        Y.tensor[ii] += vec[i] * X.tensors[i, ii]
    end
end

function dot!(vec::Vector{D}, X::PolynomialArray{D}, Y::Polynomial{D},
    k::Int, expos::ExpoTableFlat{N}) where {D, N}
    @inbounds for ii in flat_idxs(expos, k), i in 1:X.randim
        Y.tensor[ii] += vec[i] * X.tensors[i, ii]
    end
end

# dot! with SparsePartials — only iterates over non-zero axes.
function dot!(sp::SparsePartials{D}, X::PolynomialArray{D}, Y::Polynomial{D},
    k::Int, expos::ExpoTableFlat{N}) where {D, N}
    isempty(sp.indices) && return
    @inbounds for j in eachindex(sp.indices)
        i = Int(sp.indices[j])
        v = sp.values[j]
        for ii in flat_idxs(expos, k)
            Y.tensor[ii] += v * X.tensors[i, ii]
        end
    end
end

function dot!(sp::SparsePartials{D}, X::PolynomialArray{D}, Y::Polynomial{D},
    k::Tuple{Int,Int}, expos::FullExpoTableFlat{N}) where {D, N}
    isempty(sp.indices) && return
    @inbounds for j in eachindex(sp.indices)
        i = Int(sp.indices[j])
        v = sp.values[j]
        for ii in flat_idxs(expos, k)
            Y.tensor[ii] += v * X.tensors[i, ii]
        end
    end
end


# Realify : re-parametrize a PolynomialArray to real coefficients

function left_diagonal_sums(matrix::Matrix{Complex{Float64}})
    nRows, nCols = size(matrix)
    diagSums = zeros(ComplexF64, nCols + nRows - 1)
    for diag in 1:(nCols + nRows - 1)
        startCol  = min(nCols, diag)
        startRow  = max(0, diag - nCols) + 1
        nElements = min(startCol, nRows - startRow + 1)
        for j in 0:nElements-1
            diagSums[diag] += matrix[startRow + j, startCol - j]
        end
    end
    return diagSums
end

function realify(P::PolynomialArray{D}, hexpos::Vector,
    cplxEigsidxs::Vector; realEigsidxs::Vector=[],
    invMultiply::Bool=false, makeReal::Bool=false) where D

    m    = _EXPO_IDX_MAP[]
    Nc   = length(cplxEigsidxs)
    Nr   = length(realEigsidxs)
    N    = length(first(hexpos))
    realP = PolynomialArray{D}(P.order, P.ddim, P.randim)

    for axis in 1:P.randim, expoVector in hexpos
        cplxExponents = [expoVector[cplxEigsidxs[i]] for i in 1:Nc]

        cplxCombs  = []
        cplxCoeffs = []

        for i in 1:2:(Nc - 1)
            a = cplxExponents[i]
            b = cplxExponents[i+1]
            c1 = [(1/2)^(a-1) * ((-1im)^(j-1)) * binomial(a-1, j-1)
                  for j in 1:a]
            c2 = [(1/2)^(b-1) * ((1im)^(j-1)) * binomial(b-1, j-1)
                  for j in 1:b]
            push!(cplxCombs,  homog_exponents(2, a + b - 2))
            push!(cplxCoeffs, left_diagonal_sums(c1 * transpose(c2)))
        end

        src_ii = m[expoVector]

        for (prodCoeff, prodExpoVector) in
                zip(product(cplxCoeffs...), product(cplxCombs...))
            newExpoVector      = collect(expoVector)
            flatProdExpoVector = Tuple(Iterators.flatten(prodExpoVector))
            for (i, cplxidx) in enumerate(cplxEigsidxs)
                newExpoVector[cplxidx] = flatProdExpoVector[i]
            end
            tgt_ii = m[newExpoVector]
            realP.tensors[axis, tgt_ii] += reduce(*, prodCoeff) * P.tensors[axis, src_ii]
        end
    end

    if invMultiply
        CL = zeros(ComplexF64, Nc + Nr, Nc + Nr)
        newPair = true
        for i in 1:(Nc + Nr)
            if i in cplxEigsidxs && newPair
                CL[i:i+1, i:i+1] = [1/2 -1im/2; 1/2 1im/2]
                newPair = false
            elseif i in cplxEigsidxs && !newPair
                newPair = true
            elseif i in realEigsidxs
                CL[i, i] = 1
            end
        end

        coeffMatrix = zeros(ComplexF64, P.randim, length(hexpos))
        for axis in 1:P.randim, (expoIdx, expo) in enumerate(hexpos)
            coeffMatrix[axis, expoIdx] = realP.tensors[axis, m[expo]]
        end
        transformedCoeffs = inv(CL) * coeffMatrix
        for axis in 1:P.randim, (expoIdx, expo) in enumerate(hexpos)
            realP.tensors[axis, m[expo]] = transformedCoeffs[axis, expoIdx]
        end
    end

    if makeReal
        for axis in 1:P.randim, expo in hexpos
            ii = m[expo]
            realP.tensors[axis, ii] = real(realP.tensors[axis, ii])
        end
    end

    return realP
end

function realify(P::PolynomialArray{D}, cplxEigsidxs::Vector;
    realEigsidxs::Vector=[], invMultiply::Bool=false, makeReal::Bool=false) where D

    realP = PolynomialArray{D}(P.order, P.ddim, P.randim)
    for k in 1:(P.order - 1)
        hexpos = homog_exponents(P.ddim, k)
        update_poly!(realP,
            realify(P, hexpos, cplxEigsidxs;
                realEigsidxs=realEigsidxs,
                invMultiply=invMultiply, makeReal=makeReal),
            hexpos)
    end
    return realP
end
