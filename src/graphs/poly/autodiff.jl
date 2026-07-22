# integer powers at trace time: expand u^p into repeated multiplication so the
# poly graph only ever sees * nodes and needs no dedicated ^ recurrence.
function Base.:^(u::GraphBuilder{D}, p::Integer) where D
    p < 0  && error("negative integer powers not supported in graph tracing")
    p == 0 && return promote(u, one(D))[2]
    r = u
    for _ in 2:p
        r = r * u
    end
    return r
end
Base.:^(u::GraphBuilder, p::Real) =
    isinteger(p) ? u^Int(p) : error("non-integer power $p not supported")


# Sentinel coefficient types for product!

struct CoeffFwd  end   # (k - j) / k   — exp, sin recurrence
struct CoeffNeg  end   # -(k - j) / k  — cos recurrence
struct CoeffLogj end   # float(j)       — log recurrence


# update_mul! — accumulate a single convolution split of X1·X2 into Y, given that
# split's precomputed triple list (from prod_triples). update_all! uses it to add
# the DWf boundary terms incrementally without recomputing the middle splits.
@inline function update_mul!(X1::PolynomialArray, X2::PolynomialArray,
    Y::PolynomialArray, triples)
    newVecSize = X1.randim ÷ X2.randim
    @inbounds for (i1, i2, itgt) in triples
        for i in 1:newVecSize, axis in 1:X2.randim
            Y.tensors[i, itgt] +=
                X1.tensors[i + (axis - 1) * newVecSize, i1] * X2.tensors[axis, i2]
        end
    end
end


# flat-indexed typed overloads for ExpoTableFlat{N} / ConvTableFlat{N};
# these eliminate all SparseArray hash lookups in the hot inner loops

function fwd_sweep!(graph::Graph{Polynomial{D}},
    W_vec::Vector{Polynomial{D}},
    expos::ExpoTableFlat{N},
    k::Int,
    conv_table::ConvTableFlat{N}) where {D, N}

    graph.data.x          = W_vec
    graph.data.iX         = 1
    graph.data.iY         = 1
    graph.data.conv_table = conv_table

    for node in graph.ndlist
        fwd_step!(graph, node, expos, k)
    end
end

function fwd_step!(graph::Graph{Polynomial{P}},
    node::GraphNode{Polynomial{P}},
    expos::ExpoTableFlat{N},
    k::Int) where {P, N}

    op = node.operation
    op == :const && return

    nParents = length(node.parentIndices)

    if nParents >= 1
        p1 = graph.ndlist[node.parentIndices[1]]
        input1 = p1.data.extradata.noutputs == 1 ?
            p1.data.val :
            p1.data.val[p1.data.extradata.outputidx]
    end

    if nParents == 2
        p2 = graph.ndlist[node.parentIndices[2]]
        input2 = p2.data.extradata.noutputs == 1 ?
            p2.data.val :
            p2.data.val[p2.data.extradata.outputidx]
        node.func(input1, input2, node.data.val, expos, k, graph.data.conv_table)
        return
    end

    if op == :input
        node.data.val  = graph.data.x[graph.data.iX]
        graph.data.iX += 1
    elseif op == :output
        node.data.val  = input1
        graph.data.y[graph.data.iY] = node.data.val
        graph.data.iY += 1
    else
        node.func(input1, node.data.val, expos, k)
    end
end

function update_sweep!(graph::Graph{Polynomial{D}},
    W::PolynomialArray{D},
    expos::ExpoTableFlat{N},
    k::Int) where {D, N}

    for node in graph.ndlist
        if node.operation == :output || node.operation == :const
            continue
        elseif node.data.extradata.noutputs == 1
            dot!(node.data.partials, W, node.data.val, k, expos)
        elseif node.operation == :sin
            dot!(node.data.partials,             W, node.data.val[1], k, expos)
            dot!(node.data.extradata.partials,   W, node.data.val[2], k, expos)
        elseif node.operation == :cos
            dot!(node.data.extradata.partials,   W, node.data.val[1], k, expos)
            dot!(node.data.partials,             W, node.data.val[2], k, expos)
        end
    end
end

function Base.:+(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    expos::ExpoTableFlat{N}, k::Int, ::Any) where N
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = X1.tensor[ii] + X2.tensor[ii]
    end
end

function Base.:-(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    expos::ExpoTableFlat{N}, k::Int, ::Any) where N
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = X1.tensor[ii] - X2.tensor[ii]
    end
end

function Base.:-(X::Polynomial, Y::Polynomial, expos::ExpoTableFlat{N}, k::Int) where N
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = -X.tensor[ii]
    end
end

function Base.:*(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    expos::ExpoTableFlat{N}, k::Int, ct::ConvTableFlat{N}) where N
    @inbounds for (i1, i2, itgt) in ct[k]
        Y.tensor[itgt] += X1.tensor[i1] * X2.tensor[i2]
    end
end

function Base.:/(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    expos::ExpoTableFlat{N}, k::Int, ::Any) where N
    @inbounds for j in 0:k-1
        for (i1, i2, itgt) in prod_triples(expos, k, j)
            Y.tensor[itgt] += Y.tensor[i1] * X2.tensor[i2]
        end
    end
    x2_zero = X2.tensor[1]  # constant term is always at flat index 1
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = (-Y.tensor[ii] + X1.tensor[ii]) / x2_zero
    end
end

function product!(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    ::CoeffFwd, expos::ExpoTableFlat{N}, k::Int) where N
    @inbounds for j in 0:(k-1)
        c = (k - j) / k
        for (i1, i2, itgt) in prod_triples(expos, k, j)
            Y.tensor[itgt] += c * X1.tensor[i1] * X2.tensor[i2]
        end
    end
end

function product!(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    ::CoeffNeg, expos::ExpoTableFlat{N}, k::Int) where N
    @inbounds for j in 0:(k-1)
        c = -(k - j) / k
        for (i1, i2, itgt) in prod_triples(expos, k, j)
            Y.tensor[itgt] += c * X1.tensor[i1] * X2.tensor[i2]
        end
    end
end

function product!(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    ::CoeffLogj, expos::ExpoTableFlat{N}, k::Int) where N
    @inbounds for j in 0:(k-1)
        c = float(j)
        for (i1, i2, itgt) in prod_triples(expos, k, j)
            Y.tensor[itgt] += c * X1.tensor[i1] * X2.tensor[i2]
        end
    end
end

function Base.:sin(X::Polynomial, Y::Vector{<:Polynomial},
    expos::ExpoTableFlat{N}, k::Int) where N
    product!(Y[2], X, Y[1], CoeffFwd(), expos, k)
end

function Base.:cos(X::Polynomial, Y::Vector{<:Polynomial},
    expos::ExpoTableFlat{N}, k::Int) where N
    product!(Y[1], X, Y[2], CoeffNeg(), expos, k)
end

function Base.:exp(X::Polynomial, Y::Polynomial, expos::ExpoTableFlat{N}, k::Int) where N
    product!(Y, X, Y, CoeffFwd(), expos, k)
end

function Base.:log(X::Polynomial, Y::Polynomial, expos::ExpoTableFlat{N}, k::Int) where N
    product!(Y, X, Y, CoeffLogj(), expos, k)
    x_zero = X.tensor[1]
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = (-Y.tensor[ii] / k + X.tensor[ii]) / x_zero
    end
end

function dot!(X1::PolynomialArray, X2::PolynomialArray,
    Y::Polynomial, expos::ExpoTableFlat{N}, k::Int) where N
    @inbounds for j in 0:k
        for (i1, i2, itgt) in prod_triples(expos, k, j)
            for axis in 1:X1.ddim
                Y.tensor[itgt] +=
                    X1.tensors[axis, i1] * X2.tensors[axis, i2]
            end
        end
    end
end

function Base.:*(X1::PolynomialArray, X2::PolynomialArray, Y::PolynomialArray,
    expos::ExpoTableFlat{N}, k::Int, ct::ConvTableFlat{N}) where N
    newVecSize = X1.randim ÷ X2.randim
    @inbounds for (i1, i2, itgt) in ct[k]
        for i in 1:newVecSize, axis in 1:X2.randim
            Y.tensors[i, itgt] +=
                X1.tensors[i + (axis - 1) * newVecSize, i1] *
                X2.tensors[axis, i2]
        end
    end
end

# flat-indexed typed overloads for FullExpoTableFlat{N} / FullConvTableFlat{N}

function fwd_sweep!(graph::Graph{Polynomial{D}},
    W_vec::Vector{Polynomial{D}},
    expos::FullExpoTableFlat{N},
    k::Tuple{Int,Int},
    conv_table::FullConvTableFlat{N}) where {D, N}

    graph.data.x          = W_vec
    graph.data.iX         = 1
    graph.data.iY         = 1
    graph.data.conv_table = conv_table

    for node in graph.ndlist
        fwd_step!(graph, node, expos, k)
    end
end

function fwd_step!(graph::Graph{Polynomial{P}},
    node::GraphNode{Polynomial{P}},
    expos::FullExpoTableFlat{N},
    k::Tuple{Int,Int}) where {P, N}

    op = node.operation
    op == :const && return

    nParents = length(node.parentIndices)

    if nParents >= 1
        p1 = graph.ndlist[node.parentIndices[1]]
        input1 = p1.data.extradata.noutputs == 1 ?
            p1.data.val :
            p1.data.val[p1.data.extradata.outputidx]
    end

    if nParents == 2
        p2 = graph.ndlist[node.parentIndices[2]]
        input2 = p2.data.extradata.noutputs == 1 ?
            p2.data.val :
            p2.data.val[p2.data.extradata.outputidx]
        node.func(input1, input2, node.data.val, expos, k, graph.data.conv_table)
        return
    end

    if op == :input
        node.data.val  = graph.data.x[graph.data.iX]
        graph.data.iX += 1
    elseif op == :output
        node.data.val  = input1
        graph.data.y[graph.data.iY] = node.data.val
        graph.data.iY += 1
    else
        node.func(input1, node.data.val, expos, k)
    end
end

function update_sweep!(graph::Graph{Polynomial{D}},
    W::PolynomialArray{D},
    expos::FullExpoTableFlat{N},
    k::Tuple{Int,Int}) where {D, N}

    for node in graph.ndlist
        if node.operation == :output || node.operation == :const
            continue
        elseif node.data.extradata.noutputs == 1
            dot!(node.data.partials, W, node.data.val, k, expos)
        elseif node.operation == :sin
            dot!(node.data.partials,             W, node.data.val[1], k, expos)
            dot!(node.data.extradata.partials,   W, node.data.val[2], k, expos)
        elseif node.operation == :cos
            dot!(node.data.extradata.partials,   W, node.data.val[1], k, expos)
            dot!(node.data.partials,             W, node.data.val[2], k, expos)
        end
    end
end

function Base.:+(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}, ::Any) where N
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = X1.tensor[ii] + X2.tensor[ii]
    end
end

function Base.:-(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}, ::Any) where N
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = X1.tensor[ii] - X2.tensor[ii]
    end
end

function Base.:-(X::Polynomial, Y::Polynomial,
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = -X.tensor[ii]
    end
end

function Base.:*(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}, ct::FullConvTableFlat{N}) where N
    @inbounds for (i1, i2, itgt) in ct[k]
        Y.tensor[itgt] += X1.tensor[i1] * X2.tensor[i2]
    end
end

function Base.:/(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}, ::Any) where N
    ka, kn = k
    @inbounds for ja in 0:ka, jn in 0:kn
        (ja == ka && jn == kn) && continue
        for (i1, i2, itgt) in prod_triples(expos, k, (ja, jn))
            Y.tensor[itgt] += Y.tensor[i1] * X2.tensor[i2]
        end
    end
    x2_zero = X2.tensor[1]
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = (-Y.tensor[ii] + X1.tensor[ii]) / x2_zero
    end
end

function product!(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    ::CoeffFwd, expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    ka, kn = k
    total = ka + kn
    @inbounds for ja in 0:ka, jn in 0:kn
        (ja == ka && jn == kn) && continue
        c = (ka - ja + kn - jn) / total
        for (i1, i2, itgt) in prod_triples(expos, k, (ja, jn))
            Y.tensor[itgt] += c * X1.tensor[i1] * X2.tensor[i2]
        end
    end
end

function product!(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    ::CoeffNeg, expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    ka, kn = k
    total = ka + kn
    @inbounds for ja in 0:ka, jn in 0:kn
        (ja == ka && jn == kn) && continue
        c = -(ka - ja + kn - jn) / total
        for (i1, i2, itgt) in prod_triples(expos, k, (ja, jn))
            Y.tensor[itgt] += c * X1.tensor[i1] * X2.tensor[i2]
        end
    end
end

function product!(X1::Polynomial, X2::Polynomial, Y::Polynomial,
    ::CoeffLogj, expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    ka, kn = k
    @inbounds for ja in 0:ka, jn in 0:kn
        (ja == ka && jn == kn) && continue
        c = float(ja + jn)
        for (i1, i2, itgt) in prod_triples(expos, k, (ja, jn))
            Y.tensor[itgt] += c * X1.tensor[i1] * X2.tensor[i2]
        end
    end
end

function Base.:sin(X::Polynomial, Y::Vector{<:Polynomial},
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    product!(Y[2], X, Y[1], CoeffFwd(), expos, k)
end

function Base.:cos(X::Polynomial, Y::Vector{<:Polynomial},
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    product!(Y[1], X, Y[2], CoeffNeg(), expos, k)
end

function Base.:exp(X::Polynomial, Y::Polynomial,
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    product!(Y, X, Y, CoeffFwd(), expos, k)
end

function Base.:log(X::Polynomial, Y::Polynomial,
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    product!(Y, X, Y, CoeffLogj(), expos, k)
    x_zero = X.tensor[1]
    total = k[1] + k[2]
    @inbounds for ii in flat_idxs(expos, k)
        Y.tensor[ii] = (-Y.tensor[ii] / total + X.tensor[ii]) / x_zero
    end
end

function dot!(X1::PolynomialArray, X2::PolynomialArray,
    Y::Polynomial, expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    ka, kn = k
    @inbounds for ja in 0:ka, jn in 0:kn
        for (i1, i2, itgt) in prod_triples(expos, k, (ja, jn))
            for axis in 1:X1.ddim
                Y.tensor[itgt] +=
                    X1.tensors[axis, i1] * X2.tensors[axis, i2]
            end
        end
    end
end

function Base.:*(X1::PolynomialArray, X2::PolynomialArray, Y::PolynomialArray,
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}, ct::FullConvTableFlat{N}) where N
    newVecSize = X1.randim ÷ X2.randim
    @inbounds for (i1, i2, itgt) in ct[k]
        for i in 1:newVecSize, axis in 1:X2.randim
            Y.tensors[i, itgt] +=
                X1.tensors[i + (axis - 1) * newVecSize, i1] *
                X2.tensors[axis, i2]
        end
    end
end

