const POLY_BINARY_FUNC_TABLE = Dict{Symbol, Function}(
    :+ => (x1, x2, y, e, k, ct) -> Base.:+(x1, x2, y, e, k, ct),
    :- => (x1, x2, y, e, k, ct) -> Base.:-(x1, x2, y, e, k, ct),
    :* => (x1, x2, y, e, k, ct) -> Base.:*(x1, x2, y, e, k, ct),
    :/ => (x1, x2, y, e, k, ct) -> Base.:/(x1, x2, y, e, k, ct),
)

# Combined sin+cos recurrence: both polynomials live in node.data.val = [sin_poly, cos_poly]
# and both must be updated together regardless of whether the node op is :sin or :cos.
const _sincos_func = (x, y, e, k) -> begin
    Base.sin(x, y, e, k)
    Base.cos(x, y, e, k)
end

const POLY_UNARY_FUNC_TABLE = Dict{Symbol, Function}(
    :-   => (x, y, e, k) -> Base.:-(x, y, e, k),
    :exp => (x, y, e, k) -> Base.:exp(x, y, e, k),
    :log => (x, y, e, k) -> Base.:log(x, y, e, k),
    :sin => _sincos_func,
    :cos => _sincos_func,
)

function record_pgraph(F::Function,
    ddim::Int,
    rdim::Int,
    randim::Int,
    maxorder::Int,
    D::DataType)

    ngraph = record_ngraph(F, ddim, randim, D)
    fwd_evaluation_sweep!(ngraph, zeros(D, ddim))

    pgraph = Graph{Polynomial{D}}(ddim, randim)

    for i in 1:ngraph.randim
        push!(pgraph.data.y, Polynomial{D}())
    end

    for node in ngraph.ndlist
        add_poly_node!(rdim, pgraph, ngraph, node, maxorder, D)
    end

    return pgraph, ngraph
end

function add_poly_node!(
    dim::Int,
    pgraph::Graph,
    ngraph::Graph,
    node::GraphNode,
    maxorder::Int,
    D::DataType)

    op = node.operation
    nParents = length(node.parentIndices)
    u(j) = ngraph.ndlist[node.parentIndices[j]].data
    v = node.data

    # :input and :output nodes have their .val replaced in every fwd_step! call
    # (input → W_vec[iX], output → parent.val), so the initial polynomial is
    # never read. Allocate a minimal placeholder to avoid O(ddim × nterms) waste.
    if op == :input || op == :output
        placeholder = Polynomial{D}(0, 1)   # 1-entry placeholder, never accessed
        data = NodeData{Polynomial{D}}(D, placeholder, nothing, v.partials,
                                       extrapolydata(1, 1))
        func = nothing
        newNode = GraphNode{Polynomial{D}}(op, func, node.parentIndices, node.cval, data)
        push!(pgraph.ndlist, newNode)
        return
    end

    P1 = Polynomial{D}(maxorder, dim)

    if op == :sin
        P1.tensor[1] = sin(u(1).val)
        P2 = Polynomial{D}(maxorder, dim)
        P2.tensor[1] = cos(u(1).val)
        data = NodeData{Polynomial{D}}(D, [P1, P2], nothing, v.partials,
                                       extrapolydata(2, 1, v.extradata.partials))

    elseif op == :cos
        P1.tensor[1] = sin(u(1).val)
        P2 = Polynomial{D}(maxorder, dim)
        P2.tensor[1] = cos(u(1).val)
        data = NodeData{Polynomial{D}}(D, [P1, P2], nothing, v.partials,
                                       extrapolydata(2, 2, v.extradata.partials))

    else
        P1.tensor[1] = v.val
        data = NodeData{Polynomial{D}}(D, P1, nothing, v.partials, extrapolydata(1, 1))
    end

    # resolve func once here so fwd_step! never calls eval(). resolve by arity
    # first: `:-` is both binary subtraction and unary negation, so a one-parent
    # node must get the unary recurrence even though :- is also in the binary table.
    func = if nParents == 1 && op in keys(POLY_UNARY_FUNC_TABLE)
        POLY_UNARY_FUNC_TABLE[op]
    elseif op in keys(POLY_BINARY_FUNC_TABLE)
        POLY_BINARY_FUNC_TABLE[op]
    elseif op in keys(POLY_UNARY_FUNC_TABLE)
        POLY_UNARY_FUNC_TABLE[op]
    else
        nothing
    end

    newNode = GraphNode{Polynomial{D}}(op, func, node.parentIndices, node.cval, data)
    push!(pgraph.ndlist, newNode)
end
