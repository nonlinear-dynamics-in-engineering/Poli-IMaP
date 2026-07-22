# base structs and constructors

mutable struct NodeData{D}
    dataType::DataType
    val::Union{D, Vector{D}}
    bar::Any                                    # used only for reverse_AD
    partials::Any                               # was: partials::Vector
    extradata::Any                              # any extra data
end


mutable struct GraphNode{D}
    operation::Symbol
    func::Union{Nothing, Function}
    parentIndices::Vector{Int}
    cval::Any                   # constant value, only used if operation == :const
    data::NodeData{D}
end


mutable struct GraphData{D}
    dataType::DataType          # supposed to be the same as node datatype
    x::Vector{D}                # input value to graphed function
    y::Vector{D}                # output value of graphed function
    xBar::Any                   # output of reverse AD mode
    yBar::Any                   # input to reverse AD mode
    iX::Int                     # next input component to be processed
    iY::Int                     # next output component to be processed
    areBarsZero::Bool           # used to check if reverse AD mode is initialized
    conv_table::Any
end


mutable struct Graph{D}
    ndlist::Vector{GraphNode{D}}
    ddim::Int
    randim::Int
    data::GraphData{D}     # hold extra graph data not specific to node
end


struct GraphBuilder{D}
    index::Int
    graph::Graph{D}
end


const unaryOpList = [:-, :exp, :log, :sin, :cos]
const standardBinaryOpList = [:+, :-, :*, :/]
const customOpList = [:input, :output, :const]
const opStringDict = Dict(
    :- => "neg",
    :exp => "exp",
    :log => "log",
    :sin => "sin",
    :cos => "cos",
    :+ => " + ",
    :- => " - ",
    :* => " * ",
    :/ => " / ",
    :input => "inp",
    :output => "out",
    :const => "con",
)

# overload all binary operations in binaryOpList
# Uses nontrivial "promote" rules to push constants to the parent graph
# uB is any new number/symbol encountered when reading the function
# NOTE: a method D(uB) needs to exist for all uB encountered

function Base.promote(uA::GraphBuilder{D}, uB::U) where {D, U<:Union{Number, Num}}
    parentGraph = uA.graph
    prevNodes = parentGraph.ndlist
    newNodeData = deepcopy(prevNodes[uA.index].data)
    # func = nothing: :const nodes are handled by the explicit branch in fwd_step!
    newNode = GraphNode{D}(:const, nothing, Int[], D(uB), newNodeData)
    push!(prevNodes, newNode)

    return (uA, GraphBuilder{D}(length(prevNodes), parentGraph))
end

function Base.promote(uA::U, uB::GraphBuilder{D}) where {D, U<:Union{Number, Num}}
    return reverse(promote(uB, uA))
end

is_function_loaded(graph::Graph) = !isempty(graph.ndlist)

# overload all unary operations in unaryOpList
for op in unaryOpList
    @eval begin
        function Base.$op(u::GraphBuilder{D}) where {D}
            parentGraph = u.graph
            prevNodes = parentGraph.ndlist
            newNodeData = deepcopy(prevNodes[u.index].data)
            # func resolved later for poly graphs; set nothing here (num graph doesn't use it)
            newNode = GraphNode{D}(Symbol($op), nothing, [u.index], D(0), newNodeData)
            push!(prevNodes, newNode)

            return GraphBuilder{D}(length(prevNodes), parentGraph)
        end
    end
end

Base.promote(uA::GraphBuilder{D}, uB::GraphBuilder{D}) where D = (uA, uB)

for op in standardBinaryOpList
    @eval begin
        function Base.$op(uA::GraphBuilder{D}, uB::GraphBuilder{D}) where D
            parentGraph = uA.graph
            prevNodes = parentGraph.ndlist
            newNodeData = deepcopy(prevNodes[uA.index].data)
            newNode = GraphNode{D}(Symbol($op), nothing, [uA.index, uB.index], D(0), newNodeData)
            push!(prevNodes, newNode)

            return GraphBuilder{D}(length(prevNodes), parentGraph)
        end
    end
end

for op in standardBinaryOpList
    @eval begin
        function Base.$op(uA::GraphBuilder{D}, uB::U) where {D, U<:Union{Number, Num}}
            return $op(promote(uA, uB)...)
        end

        function Base.$op(uA::U, uB::GraphBuilder{D}) where {D, U<:Union{Number, Num}}
            return $op(promote(uA, uB)...)
        end
    end
end


# print stuff
function Base.show(io::IO, node::GraphNode)
    parents = node.parentIndices
    nParents = length(parents)

    opString = opStringDict[node.operation]

    if nParents <= 2
        oneParent(i::Int) = (nParents < i) ? "   " : @sprintf "%-3d" parents[i]
        parentString = oneParent(1) * "  " * oneParent(2)
    else
        parentString = string(parents)
    end

    if node.operation == :const
        if node.data.dataType <: Union{Num, Complex{Num}}
            dataString = @sprintf "const: %s" string(node.cval)
        else
            dataString = @sprintf "const: % .3e + % .3eim" real(node.cval) imag(node.cval)
        end
    else
        dataString = string(node.data)
    end

    return print(io, opString, " | ", parentString, " | ", dataString)
end


function Base.show(io::IO, graph::Graph)
    return begin
        println(io, " Computational graph:\n")
        println(io, " index | op  | parents  | data")
        println(io, " ------------------------------")
        for (i, node) in enumerate(graph.ndlist)
            @printf io "   %3d | " i
            println(io, node)
        end
    end
end


function Base.show(io::IO, n::NodeData)
    if n.dataType == Float64
        @printf(io, "val: %.3e,   bar: %.3e", n.val, n.bar)

    elseif n.dataType == ComplexF64
        @printf(io, "val: %.3e + % .3eim,   bar: %.3e + % .3eim",
        real(n.val), imag(n.val), real(n.bar), imag(n.bar))

    elseif n.dataType == Num || n.dataType == Complex{Num}
        @printf(io, "val: %s,   bar: %s", string(n.val), string(n.bar))
    end

end
