function parametrize_nonautonomous!(sys::System, pset::ParSettings)
    load_linear_coeffs!(sys, pset, option="nonautonomous")
    sys.W_vec = array_to_vec(sys.W)
    _parametrize_nonautonomous_typed!(sys, pset, Val(pset.rdim))
end

function _parametrize_nonautonomous_typed!(sys::System, pset::ParSettings, ::Val{N}) where N
    fullexpos = pset.nonautexpos_flat::FullExpoTableFlat{N}
    fullconv  = pset.nonautconv_flat::FullConvTableFlat{N}

    # Initialize at (0, 1) — the linear nonautonomous terms
    ∇!(sys.W, sys.DW, fullexpos, (0, 1))
    *(sys.DW, sys.f, sys.DWf, fullexpos, (0, 1), fullconv)
    fwd_sweep!(sys.gF, sys.W_vec, fullexpos, (0, 1), fullconv)

    if !sys.B_is_const
        # Extend the B(W) series into the nonautonomous bi-degrees. The
        # autonomous (ka, 0) coefficients in gB.data.y already persist from
        # parametrize_autonomous!.
        fwd_sweep!(sys.gB, sys.W_vec, fullexpos, (0, 1), fullconv)
    end

    print("\nStarting nonautonomous parametrization.")
    p = Progress(pset.knonaut * pset.kaut; barglyphs=BarGlyphs("[=> ]"), barlen=20)

    for kn = 1:pset.knonaut
        for ka = 0:(pset.kaut - 1)
            k = (ka, kn)
            k == (0, 1) && continue  # already initialized above

            fwd_sweep!(sys.gF, sys.W_vec, fullexpos, k, fullconv)
            *(sys.DW, sys.f, sys.DWf, fullexpos, k, fullconv)

            e1 = E1(sys, fullexpos, k)
            e2 = E2(sys, fullexpos, k, fullconv)
            e3 = E3(sys, pset, fullexpos, k)
            solve_homological_equations!(sys, pset, fullexpos[k], e1 - e2 - e3)

            update_all!(sys, pset, fullexpos, k, fullconv)

            next!(p)
        end
    end
    print("\nNonautonomous parametrization complete.\n")
end
