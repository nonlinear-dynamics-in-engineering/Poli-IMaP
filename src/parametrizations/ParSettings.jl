Base.@kwdef mutable struct ParSettings
    autidxs::Vector{Int}
    auttgaxes::Vector{Int}
    tgaxes::Vector{Int}
    parstyle::String
    kaut::Int
    rdim::Int = 0
    autdim::Int = 0
    fullspectrum::Bool = true
    resonances::Dict{Int, Tuple} = Dict{Int, Tuple}()
    nonautidxs::Union{Vector{Int}, Nothing} = nothing
    nonauttgaxes::Union{Vector{Int}, Nothing} = nothing
    statpidxs::Union{Vector{Int}, Nothing} = nothing
    knonaut::Union{Int, Nothing} = nothing
    kstatp::Union{Int, Nothing} = nothing
    expos::Dict{Any, Vector} = Dict{Any, Vector}()
    autexpos::Dict{Int, Vector} = Dict{Int, Vector}()
    nonautexpos::Dict{Int, Vector} = Dict{Int, Vector}()
    resexpos::Dict{Int, Vector} = Dict{Int, Vector}()
    normaxes::Vector{Int} = []
    rdaxes::Vector{Int} = []
    autrdaxes::Vector{Int} = []
    nonautrdaxes::Vector{Int} = []
    nonautmap::Dict = Dict()
    internal_res_tol::Float64 = .0001
    cross_res_tol::Float64 = .0001
    # Flat-indexed versions: Vector/Matrix storage with Int32 flat indices
    autexpos_flat::Any = nothing
    autconv_flat::Any = nothing
    nonautexpos_flat::Any = nothing
    nonautconv_flat::Any = nothing
end

function set_parametrization_settings(sys::System; kwargs...)
    pset = ParSettings(; kwargs...)
    complete_fields!(sys, pset)
    generate_exponents!(pset)
    generate_resonant_exponents!(pset)
    return pset
end

function complete_fields!(sys::System, pset::ParSettings)
    pset.rdim      = length(pset.tgaxes)
    pset.rdaxes    = collect(1:pset.rdim)
    pset.normaxes  = setdiff(collect(1:sys.ddim), pset.tgaxes)
    pset.autdim    = length(pset.auttgaxes)
    pset.autrdaxes = [i for i in 1:length(pset.auttgaxes)]
    if !isnothing(pset.nonautidxs)
        pset.nonautrdaxes = [i for i in length(pset.auttgaxes) + 1:length(pset.auttgaxes) + length(pset.nonauttgaxes)]
        pset.nonautmap = Dict((pset.nonautidxs[i - length(pset.auttgaxes)],
        pset.nonautidxs[i + 1 - length(pset.auttgaxes)]) => (i, i + 1)
        for i in length(pset.auttgaxes) + 1:2:length(pset.auttgaxes) + length(pset.nonauttgaxes))
    end
end

function generate_exponents!(pset::ParSettings)
    expos    = Dict{Any, Vector}()
    autexpos = Dict{Int, Vector}()

    if isnothing(pset.knonaut) == false
        nonautexpos = Dict{Int, Vector}()

        for kaut in 0:pset.kaut
            autexpos[kaut] =
            homog_exponents([length(pset.auttgaxes), length(pset.nonauttgaxes)], [kaut, 0])
        end
        for knonaut in 0:pset.knonaut
            nonautexpos[knonaut] =
            homog_exponents([length(pset.auttgaxes), length(pset.nonauttgaxes)], [0, knonaut])
        end
        for kaut in 0:pset.kaut, knonaut in 0:pset.knonaut
            expos[(kaut, knonaut)] =
            homog_exponents([length(pset.auttgaxes), length(pset.nonauttgaxes)], [kaut, knonaut])
        end
        pset.autexpos    = autexpos
        pset.nonautexpos = nonautexpos

    else
        for kaut in 0:pset.kaut
            autexpos[kaut] = homog_exponents(length(pset.auttgaxes), kaut)
            expos[kaut]    = homog_exponents(length(pset.auttgaxes), kaut)
        end
        pset.autexpos = autexpos
    end

    pset.expos = expos

    # Build flat-indexed tables for maximum performance
    etf = _build_expo_table_flat(pset.autexpos, pset.kaut, Val(pset.rdim))
    pset.autexpos_flat      = etf
    pset.autconv_flat       = _build_conv_table_flat(etf, pset.kaut)
    # Set module-level map so non-hot utility functions can resolve flat indices
    _EXPO_IDX_MAP[] = etf.expo_to_idx

    if !isnothing(pset.knonaut)
        fetf = _build_full_expo_table_flat(pset.expos, pset.kaut, pset.knonaut, Val(pset.rdim))
        pset.nonautexpos_flat  = fetf
        pset.nonautconv_flat   = _build_full_conv_table_flat(fetf, pset.kaut, pset.knonaut)
    end
end

function _build_expo_table_flat(autexpos::Dict, kaut::Int, ::Val{N}) where N
    expo_to_idx = Dict{Vector{Int}, Int32}()
    expos_dict  = Dict{Int, Vector{Vector{Int}}}()
    idxs_dict   = Dict{Int, Vector{Int32}}()
    prod_dict   = Dict{Tuple{Int,Int}, Vector{Tuple{Int32,Int32,Int32}}}()
    idx = Int32(1)
    for k in 0:kaut
        ev = [Vector{Int}(e) for e in autexpos[k]]
        expos_dict[k] = ev
        for e in ev
            expo_to_idx[e] = idx
            idx += 1
        end
    end
    for k in 0:kaut
        idxs_dict[k] = [expo_to_idx[e] for e in expos_dict[k]]
    end
    nterms = Int(idx) - 1
    rdim = isempty(expos_dict[0]) ? N : length(first(expos_dict[0]))
    tgt_buf = Vector{Int}(undef, rdim)
    for k in 1:kaut
        for j in 0:k
            triples = Tuple{Int32,Int32,Int32}[]
            for e1 in expos_dict[j], e2 in expos_dict[k-j]
                @. tgt_buf = e1 + e2 - 1
                push!(triples, (expo_to_idx[e1], expo_to_idx[e2], expo_to_idx[tgt_buf]))
            end
            prod_dict[(k, j)] = triples
        end
    end
    return ExpoTableFlat{N}(expos_dict, idxs_dict, expo_to_idx, nterms, prod_dict)
end

function _build_conv_table_flat(et::ExpoTableFlat{N}, kaut::Int) where N
    conv = Dict{Int, Vector{Tuple{Int32,Int32,Int32}}}()
    rdim = isempty(et.expos[0]) ? N : length(first(et.expos[0]))
    tgt_buf = Vector{Int}(undef, rdim)
    for k in 1:kaut
        triples = Tuple{Int32,Int32,Int32}[]
        for j in 0:k, e1 in et.expos[j], e2 in et.expos[k-j]
            @. tgt_buf = e1 + e2 - 1
            push!(triples, (et.expo_to_idx[e1], et.expo_to_idx[e2], et.expo_to_idx[tgt_buf]))
        end
        conv[k] = triples
    end
    return ConvTableFlat{N}(conv)
end

function _build_full_expo_table_flat(expos::Dict, kaut::Int, knonaut::Int, ::Val{N}) where N
    aut_m = _EXPO_IDX_MAP[]
    isnothing(aut_m) && error("_EXPO_IDX_MAP must be set before building FullExpoTableFlat")
    m = copy(aut_m)
    idx = Int32(length(m) + 1)

    expos_dict  = Dict{Tuple{Int,Int}, Vector{Vector{Int}}}()
    idxs_dict   = Dict{Tuple{Int,Int}, Vector{Int32}}()
    prod_dict   = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}}, Vector{Tuple{Int32,Int32,Int32}}}()

    # First pass: register all exponents, assigning new indices to unseen ones
    for ka in 0:kaut, kn in 0:knonaut
        ev = [Vector{Int}(e) for e in expos[(ka, kn)]]
        expos_dict[(ka, kn)] = ev
        for e in ev
            if !haskey(m, e)
                m[e] = idx
                idx += 1
            end
        end
    end

    # Update global map so utility functions see all exponents
    _EXPO_IDX_MAP[] = m

    # Second pass: build flat index lists
    for ka in 0:kaut, kn in 0:knonaut
        idxs_dict[(ka, kn)] = [m[e] for e in expos_dict[(ka, kn)]]
    end

    nterms = Int(idx) - 1

    rdim = isempty(expos_dict[(0,0)]) ? N : length(first(expos_dict[(0,0)]))
    tgt_buf = Vector{Int}(undef, rdim)

    # Build product triples grouped by (k, j)
    for ka in 0:kaut, kn in 0:knonaut
        (ka == 0 && kn == 0) && continue
        k = (ka, kn)
        for ja in 0:ka, jn in 0:kn
            j = (ja, jn)
            triples = Tuple{Int32,Int32,Int32}[]
            for e1 in expos_dict[(ja, jn)], e2 in expos_dict[(ka-ja, kn-jn)]
                @. tgt_buf = e1 + e2 - 1
                push!(triples, (m[e1], m[e2], m[tgt_buf]))
            end
            prod_dict[(k, j)] = triples
        end
    end
    return FullExpoTableFlat{N}(expos_dict, idxs_dict, m, nterms, prod_dict)
end

function _build_full_conv_table_flat(et::FullExpoTableFlat{N}, kaut::Int, knonaut::Int) where N
    conv = Dict{Tuple{Int,Int}, Vector{Tuple{Int32,Int32,Int32}}}()
    rdim = isempty(et.expos[(0,0)]) ? N : length(first(et.expos[(0,0)]))
    tgt_buf = Vector{Int}(undef, rdim)
    for ka in 0:kaut, kn in 0:knonaut
        (ka == 0 && kn == 0) && continue
        k = (ka, kn)
        triples = Tuple{Int32,Int32,Int32}[]
        for ja in 0:ka, jn in 0:kn
            for e1 in et.expos[(ja,jn)], e2 in et.expos[(ka-ja,kn-jn)]
                @. tgt_buf = e1 + e2 - 1
                push!(triples, (et.expo_to_idx[e1], et.expo_to_idx[e2], et.expo_to_idx[tgt_buf]))
            end
        end
        conv[k] = triples
    end
    return FullConvTableFlat{N}(conv)
end

function generate_resonant_exponents!(pset::ParSettings)
    resexpos = Dict{Int, Vector}(i => [] for i in 1:pset.rdim)

    for orders in keys(pset.expos),
        expo in pset.expos[orders],
        rdaxis in keys(pset.resonances)

            if 1 == dot(pset.resonances[rdaxis], expo .- 1)
                push!(resexpos[rdaxis], expo)
            end
    end

    pset.resexpos = resexpos
end
