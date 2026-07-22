# Flat-indexed overloads
function E1(sys::System, expos::ExpoTableFlat{N}, k::Int) where N
    return homog_components(sys.gF.data.y, expos[k])
end

function E2(sys::System, expos::ExpoTableFlat{N}, k::Int,
           conv::ConvTableFlat{N}) where N
    if sys.B_is_const
        return homog_components(sys.DWf, expos[k])
    else
        # E2 = [B(W) · DW · f]_k via the row-map matrix·vector product (dense or
        # sparse). BDWf order-k slots are zero before this call (written once per
        # order), so the accumulating product yields exactly the order-k terms.
        bmatvec!(sys.BDWf, sys.gB.data.y, sys.DWf, sys.B_rowmap, k, conv)
        return homog_components(sys.BDWf, expos[k])
    end
end

function E1(sys::System, expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    return homog_components(sys.gF.data.y, expos[k])
end

function E2(sys::System, expos::FullExpoTableFlat{N}, k::Tuple{Int,Int},
           conv::FullConvTableFlat{N}) where N
    if sys.B_is_const
        return homog_components(sys.DWf, expos[k])
    else
        bmatvec!(sys.BDWf, sys.gB.data.y, sys.DWf, sys.B_rowmap, k, conv)
        return homog_components(sys.BDWf, expos[k])
    end
end

function _compute_E3Poly!(E3Poly, sys, pset, hexpos)
    m       = _EXPO_IDX_MAP[]
    rdexpo  = ones(Int, pset.rdim)   # reused across rdaxis iterations
    newexpo = Vector{Int}(undef, pset.rdim)  # reused across inner loop

    for rdaxis in pset.nonautrdaxes
        rdexpo[rdaxis] = 2
        rdidx = m[rdexpo]  # flat index of the linear nonautonomous exponent
        rdexpo[rdaxis] = 1  # restore for next rdaxis

        for expo in hexpos
            expo[rdaxis] == 1 && continue
            src_ii = m[expo]

            for axis in 1:pset.autdim, autaxis in pset.autrdaxes
                newexpo .= expo
                newexpo[autaxis] += 1
                newexpo[rdaxis]  -= 1
                coeff = expo[autaxis] - 1   # = newexpo[autaxis] - 1 before increment
                tgt_ii = m[newexpo]
                E3Poly[axis].tensor[src_ii] += coeff *
                    sys.W.tensors[axis, tgt_ii] *
                    sys.f.tensors[autaxis, rdidx]
            end
        end
    end
end

function E3(sys::System, pset::ParSettings, expos::ExpoTableFlat{N}, k::Int) where N
    hexpos  = expos[k]
    E3Poly  = [Polynomial{sys.D}(pset.kaut, pset.rdim) for _ in 1:pset.autdim]
    _compute_E3Poly!(E3Poly, sys, pset, hexpos)

    E3result = SparseArray(zeros(sys.D, sys.ddim, length(hexpos)))
    hc = homog_components(E3Poly, hexpos)  # autdim × nMonomials
    if !sys.B_is_const
        E3result[pset.auttgaxes, :] = sys.B₀[pset.auttgaxes, pset.auttgaxes] * hc
    else
        E3result[pset.auttgaxes, :] = hc
    end
    return E3result
end

function E3(sys::System, pset::ParSettings, expos::FullExpoTableFlat{N}, k::Tuple{Int,Int}) where N
    hexpos  = expos[k]
    E3Poly  = [Polynomial{sys.D}(pset.kaut + pset.knonaut, pset.rdim) for _ in 1:pset.autdim]
    _compute_E3Poly!(E3Poly, sys, pset, hexpos)

    E3result = SparseArray(zeros(sys.D, sys.ddim, length(hexpos)))
    hc = homog_components(E3Poly, hexpos)
    if !sys.B_is_const
        E3result[pset.auttgaxes, :] = sys.B₀[pset.auttgaxes, pset.auttgaxes] * hc
    else
        E3result[pset.auttgaxes, :] = hc
    end
    return E3result
end
