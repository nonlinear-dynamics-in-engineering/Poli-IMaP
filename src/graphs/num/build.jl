# create a graph of a provided function
function record_ngraph(
    F::Function,
    ddim::Int,
    randim::Int,
    D::DataType)

    # same dataType for both Graph and Nodes data types:
    graph = Graph{D}(ddim, randim)
    load_function!(F, graph, NodeData{D}(ddim))
    graph.data.areBarsZero = false
    return graph
end

# load a function using Operator Overloading, and store its computational graph
function load_function!(
    f::Function,
    graph::Graph{D},
    inputData::NodeData{D}) where D <: NumTypes

    empty!(graph.ndlist)

    # push new nodes for function inputs
    xGB = [GraphBuilder{D}(i, graph) for i=1:(graph.ddim)]
    for xComp in xGB
        # func = nothing: :input nodes handled explicitly in fwd_step!
        inputNode = GraphNode{D}(:input, nothing, Int[], D(0), deepcopy(inputData))
        push!(graph.ndlist, inputNode)
    end

    # push new nodes for all intermediate operations, using Operator Overloading
    yGB = f(xGB)
    if !(yGB isa Vector)
        yGB = [yGB]
    end

    # push new nodes for function outputs
    for yComp in yGB
        # A component may come back as a plain constant (no input dependence),
        # e.g. when F is generated from a symbolic expression that reduces a row
        # to a constant. Emit a :const node and point the output at it.
        if yComp isa GraphBuilder
            parentIdx = yComp.index
        else
            constNode = GraphNode{D}(:const, nothing, Int[], D(yComp), deepcopy(inputData))
            push!(graph.ndlist, constNode)
            parentIdx = length(graph.ndlist)
        end
        outputData = deepcopy(inputData)
        # func = nothing: :output nodes handled explicitly in fwd_step!
        outputNode = GraphNode{D}(:output, nothing, [parentIdx], D(0), outputData)
        push!(graph.ndlist, outputNode)
    end
end
