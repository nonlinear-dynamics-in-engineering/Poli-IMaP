# Polynomials node data constructors
mutable struct extrapolydata
    noutputs::Int
    outputidx::Int
    partials::Any    # was: Union{Vector, Nothing}
end

extrapolydata(n::Int, m::Int) = extrapolydata(n, m, nothing)

NodeData{Polynomial{D}}() where D =
NodeData{Polynomial{D}}(Polynomial{D},              # .dataType
                        Polynomial{D}(),            # .val
                        D(0),                       # .bar
                        D[],                        # .partials
                        extrapolydata(1, 1)         # .extradata
)

GraphNode{Polynomial{D}}(op::Symbol, i::Vector{Int}, data::NodeData{Polynomial{D}}) where D =
GraphNode{D}(op, nothing, i, D(0), data)

# Polynomials graph data constructors
GraphData{Polynomial{D}}() where D =
GraphData{Polynomial{D}}(Polynomial{D},          # .dataType
                         Polynomial{D}[],        # .x
                         Polynomial{D}[],        # .y
                         nothing,                # .xBar
                         nothing,                # .yBar
                         1,                      # iX
                         1,                      # iY
                         false,                  # .areBarsZero
                         nothing                 # .conv_table — set by fwd_sweep! each call
)


Graph{Polynomial{D}}() where D =
Graph{Polynomial{D}}([], 1, 1, GraphData{Polynomial{D}}())

Graph{Polynomial{D}}(n::Int, m::Int) where D =
Graph{Polynomial{D}}([], n, m, GraphData{Polynomial{D}}())
