# this file's node value types
const NumTypes = Union{Float64, Complex{Float64}, Num, Complex{Num}}

mutable struct extradata
    partials::Any   
end

extradata() = extradata(nothing)

function NodeData{D}(ddim::Int) where D <: NumTypes
    try
        return NodeData{D}(D,
                           D(0),
                           D(0),
                           SparsePartials{D}(),   
                           extradata())
    catch
        error("Variable type not supported")
    end
end

function GraphNode{D}(op::Symbol, i::Vector{Int}, data::NodeData{D}) where D <: NumTypes
    try
        return GraphNode{D}(op, nothing, i, D(0), data)
    catch
        error("Variable type not supported")
    end
end

GraphData{D}() where D <: NumTypes = GraphData(D, D[], D[], D[], D[], 1, 1, false, nothing)

function Graph{D}() where D <: NumTypes
    return Graph{D}(GraphNode{D}[], 1, 1, GraphData{D}())
end

function Graph{D}(n::Int, m::Int) where D <: NumTypes
    return Graph{D}(GraphNode{D}[], n, m, GraphData{D}())
end
