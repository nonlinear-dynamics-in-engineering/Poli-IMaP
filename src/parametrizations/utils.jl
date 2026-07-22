function load_linear_coeffs!(sys::System, pset::ParSettings; option::String)

    if option == "autonomous"
        m = _EXPO_IDX_MAP[]
        for ax in 1:sys.ddim, (expo, tgax) in zip(pset.autexpos[1], pset.auttgaxes)
            sys.W.tensors[ax, m[expo]] = sys.r_eigvecs[ax, tgax]
        end

        # f rows are indexed by the reduced axes (like the nonautonomous branch
        # below), not by the eigenvalue indices: the two only coincide when the
        # target modes happen to be the first in the eigenvalue ordering
        for (expo, tgax, rdax) in
            zip(pset.autexpos[1], pset.auttgaxes, pset.autrdaxes)
            sys.f.tensors[rdax, m[expo]] = sys.eigvals[tgax]
        end

    elseif option == "nonautonomous"
        m = _EXPO_IDX_MAP[]
        for (ax1, ax2) in keys(pset.nonautmap)
            expoReal = ones(Int, pset.rdim)
            expoReal[pset.nonautmap[(ax1, ax2)][1]] = 2
            sys.W.tensors[ax1, m[expoReal]] = sys.D(0.5)
            sys.W.tensors[ax2, m[expoReal]] = sys.D(0.5im)
            expoImag = ones(Int, pset.rdim)
            expoImag[pset.nonautmap[(ax1, ax2)][2]] = 2
            sys.W.tensors[ax1, m[expoImag]] = sys.D(0.5)
            sys.W.tensors[ax2, m[expoImag]] = sys.D(-0.5im)
        end

        for (expo, ax, rdax) in
            zip(pset.nonautexpos[1], pset.nonauttgaxes, pset.nonautrdaxes)
            sys.f.tensors[rdax, m[expo]] = sys.eigvals[ax]
        end
    end
end

# zero the order-k coefficients of a PolynomialArray 
@inline function _zero_order!(P::PolynomialArray, idxs)
    @inbounds for ii in idxs, ax in 1:P.randim
        P.tensors[ax, ii] = 0
    end
end

# add the listed coefficients of src into dst 
@inline function _add_order!(dst::PolynomialArray, src::PolynomialArray, idxs)
    @inbounds for ii in idxs, ax in 1:dst.randim
        dst.tensors[ax, ii] += src.tensors[ax, ii]
    end
end

# Flat-indexed typed overload: ExpoTableFlat{N} / ConvTableFlat{N}
function update_all!(sys::System, ::ParSettings,
    expos::ExpoTableFlat{N}, k::Int, conv::ConvTableFlat{N}) where N

    update_sweep!(sys.gF, sys.W, expos, k)
    update_vec!(sys.W_vec, sys.W, expos, k)
    ∇!(sys.W, sys.DW, expos, k)

    update_mul!(sys.DW, sys.f, sys.DWf, prod_triples(expos, k, 0))
    update_mul!(sys.DW, sys.f, sys.DWf, prod_triples(expos, k, k - 1))

    if !sys.B_is_const
        fwd_sweep!(sys.gB, sys.W_vec, expos, k, conv)
    end
end

# Flat-indexed nonautonomous typed overload: FullExpoTableFlat{N} / FullConvTableFlat{N}
function update_all!(sys::System, ::ParSettings,
    expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}, conv::FullConvTableFlat{N}) where N
    ka, kn = k

    update_sweep!(sys.gF, sys.W, expos, k)
    update_vec!(sys.W_vec, sys.W, expos, k)

    gidx = Int32[]
    ka ≥ 1 && append!(gidx, flat_idxs(expos, (ka - 1, kn)))
    kn ≥ 1 && append!(gidx, flat_idxs(expos, (ka, kn - 1)))
    _zero_order!(sys.DWk, gidx)
    ∇!(sys.W, sys.DWk, expos, k)

    update_mul!(sys.DW,  sys.f, sys.DWf, prod_triples(expos, k, (0, 0)))
    ka ≥ 1 && update_mul!(sys.DWk, sys.f, sys.DWf, prod_triples(expos, k, (ka - 1, kn)))
    kn ≥ 1 && update_mul!(sys.DWk, sys.f, sys.DWf, prod_triples(expos, k, (ka, kn - 1)))

    # add the isolated ∇Wₖ into the running DW for higher orders
    _add_order!(sys.DW, sys.DWk, gidx)

    if !sys.B_is_const
        fwd_sweep!(sys.gB, sys.W_vec, expos, k, conv)
    end
end
