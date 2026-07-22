# forward sweep through graph, to compute each node.data.val
# and initialize each node.data.bar
function fwd_evaluation_sweep!(
    graph::Graph,
    x::Vector{D}
) where D
    t = graph.data

    (is_function_loaded(graph)) ||
        throw(DomainError("graph: hasn't been loaded with a function"))
    (length(x) == graph.ddim) ||
        throw(DomainError("x: # components doesn't match graph's ddim"))

    t.x  = x
    t.y  = zeros(D, graph.randim)
    t.iX = 1
    t.iY = 1

    for node in graph.ndlist
        fwd_evaluation_step!(graph, node)
    end
    t.areBarsZero = true
end


function fwd_evaluation_step!(
    graph::Graph,
    node::GraphNode
)
    op       = node.operation
    nParents = length(node.parentIndices)
    u(j)     = graph.ndlist[node.parentIndices[j]].data
    v        = node.data
    t        = graph.data

    if op == :input
        v.val = t.x[t.iX]
        t.iX += 1

    elseif op == :output
        v.val = u(1).val
        t.y[t.iY] = v.val
        t.iY += 1

    elseif op == :const
        v.val = node.cval

    elseif nParents == 1
        v.val = eval(op)(u(1).val)

    elseif nParents == 2
        v.val = eval(op)(u(1).val, u(2).val)

    else
        throw(DomainError("unsupported elemental operation: " * String(op)))
    end

    # Use the node's own datatype, not a global `D`.
    v.bar = v.dataType(0)
end


# fwd sweep through graph to compute node.data.partials
function fwd_adjoint_sweep!(
    graph::Graph,
    x::Vector{D},
) where D
    t    = graph.data
    t.x  = x
    t.iX = 1
    t.iY = 1

    for node in graph.ndlist
        fwd_adjoint_step!(graph, node, D)  # D is a Type{D} here — type-stable dispatch
    end
end


function fwd_adjoint_step!(
    graph::Graph,
    node::GraphNode,
    ::Type{P}       # P is a compile-time type parameter — enables type-stable dispatch
) where P
    op       = node.operation
    nParents = length(node.parentIndices)
    u(j)     = graph.ndlist[node.parentIndices[j]].data
    v        = node.data
    t        = graph.data

    if op == :input
        v.partials = SparsePartials{P}(Int32[Int32(t.iX)], P[P(1)])
        v.val = t.x[t.iX]
        t.iX += 1

    elseif op == :output
        v.partials = u(1).partials
        t.iY += 1

    elseif op == :const
        # do nothing

    elseif nParents == 1
        # P is a compile-time type parameter, so sp1 is inferred as SparsePartials{P}
        sp1 = u(1).partials::SparsePartials{P}
        u1val = u(1).val
        if op == :-
            v.partials = SparsePartials{P}(sp1.indices, _scale(P(-1), sp1.values))
        elseif op == :exp
            v.partials = SparsePartials{P}(sp1.indices, _scale(exp(u1val), sp1.values))
        elseif op == :log
            v.partials = SparsePartials{P}(sp1.indices, _scale(P(1)/u1val, sp1.values))
        elseif op == :sin
            v.partials           = SparsePartials{P}(sp1.indices, _scale( cos(u1val), sp1.values))
            v.extradata.partials = SparsePartials{P}(sp1.indices, _scale(-sin(u1val), sp1.values))
        elseif op == :cos
            v.partials           = SparsePartials{P}(sp1.indices, _scale(-sin(u1val), sp1.values))
            v.extradata.partials = SparsePartials{P}(sp1.indices, _scale( cos(u1val), sp1.values))
        end

    elseif nParents == 2
        sp1 = u(1).partials::SparsePartials{P}
        sp2 = u(2).partials::SparsePartials{P}
        u1val, u2val = u(1).val, u(2).val
        if op == :+
            v.partials = _sparse_add(sp1, sp2)
        elseif op == :-
            v.partials = _sparse_sub(sp1, sp2)
        elseif op == :*
            # ∂(u1*u2) = u1*∂u2 + u2*∂u1  — build scaled copies then merge
            v.partials = _sparse_add_scaled(u1val, sp2, u2val, sp1)
        elseif op == :/
            b = u2val
            # ∂(u1/u2) = ∂u1/u2 - u1/u2² * ∂u2
            v.partials = _sparse_add_scaled(P(1)/b, sp1, -u1val/(b*b), sp2)
        end
    end
end


# carry out reverse AD mode, using a forward evaluation sweep
# then a reverse adjoint sweep through a function's computational graph
function reverse_AD!(
    graph::Graph,
    x::Vector{D},
    yBar::Vector{D}
) where D
    t = graph.data
    fwd_evaluation_sweep!(graph, x)
    rev_adjoint_sweep!(graph, yBar)
    return t.y, t.xBar
end


# reverse sweep through graph, to evaluate each node.data.bar
function rev_adjoint_sweep!(
    graph::Graph,
    yBar::Vector{D}
) where D
    t = graph.data

    (is_function_loaded(graph)) ||
        throw(DomainError("graph: hasn't been loaded with a function"))
    (length(yBar) == graph.randim) ||
        throw(DomainError("yBar: # components doesn't match graph's randim"))

    t.xBar = zeros(D, graph.ddim)
    t.yBar = yBar
    t.iX   = graph.ddim
    t.iY   = graph.randim
    if !(t.areBarsZero)
        for node in graph.ndlist
            node.data.bar = D(0)
        end
    end

    for node in Iterators.reverse(graph.ndlist)
        rev_adjoint_step!(graph, node)
    end
    t.areBarsZero = false
end


function rev_adjoint_step!(
    graph::Graph,
    node::GraphNode
)
    op       = node.operation
    nParents = length(node.parentIndices)
    u(j)     = graph.ndlist[node.parentIndices[j]].data
    v        = node.data
    t        = graph.data

    if op == :input
        t.xBar[t.iX] = v.bar
        t.iX -= 1

    elseif op == :output
        v.bar = t.yBar[t.iY]
        t.iY -= 1
        u(1).bar += v.bar

    elseif op == :const
        # no parent nodes; do nothing

    elseif nParents == 1
        if op == :-
            u(1).bar -= v.bar
        elseif op == :exp
            u(1).bar += v.bar * v.val
        elseif op == :log
            u(1).bar += v.bar / u(1).val
        elseif op == :sin
            u(1).bar += v.bar * cos(u(1).val)
        elseif op == :cos
            u(1).bar -= v.bar * sin(u(1).val)
        else
            throw(DomainError("unsupported elemental operation: " * String(op)))
        end

    elseif nParents == 2
        if op == :+
            u(1).bar += v.bar
            u(2).bar += v.bar
        elseif op == :-
            u(1).bar += v.bar
            u(2).bar -= v.bar
        elseif op == :*
            u(1).bar += v.bar * u(2).val
            u(2).bar += v.bar * u(1).val
        elseif op == :/
            u(1).bar += v.bar / u(2).val
            u(2).bar -= v.bar * u(1).val / ((u(2).val)^2)
        else
            throw(DomainError("unsupported elemental operation: " * String(op)))
        end
    else
        throw(DomainError("unsupported elemental operation: " * String(op)))
    end
end
