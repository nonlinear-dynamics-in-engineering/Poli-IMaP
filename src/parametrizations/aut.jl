function parametrize_autonomous!(sys::System, pset::ParSettings)
    _compute_jacobian!(sys)   # populate gF node partials needed by update_sweep!
    load_linear_coeffs!(sys, pset, option="autonomous")
    sys.W_vec = array_to_vec(sys.W)
    _parametrize_autonomous_typed!(sys, pset, Val(pset.rdim))
end

function _parametrize_autonomous_typed!(sys::System, pset::ParSettings, ::Val{N}) where N
    # Type assertions: pset fields are Any, but we assert the concrete type here
    # so the rest of the function body is fully type-stable.
    autexpos = pset.autexpos_flat::ExpoTableFlat{N}
    autconv  = pset.autconv_flat::ConvTableFlat{N}

    ∇!(sys.W, sys.DW, autexpos, 1)
    *(sys.DW, sys.f, sys.DWf, autexpos, 1, autconv)
    fwd_sweep!(sys.gF, sys.W_vec, autexpos, 1, autconv)

    if !sys.B_is_const
        fwd_sweep!(sys.gB, sys.W_vec, autexpos, 1, autconv)
    end

    print("Starting autonomous parametrization.")
    p = Progress(pset.kaut - 1; barglyphs=BarGlyphs("[=> ]"), barlen=20)
    for k = 2:(pset.kaut - 1)

        fwd_sweep!(sys.gF, sys.W_vec, autexpos, k, autconv)
        *(sys.DW, sys.f, sys.DWf, autexpos, k, autconv)

        e1 = E1(sys, autexpos, k)
        e2 = E2(sys, autexpos, k, autconv)
        solve_homological_equations!(sys, pset, autexpos[k], e1 - e2)

        update_all!(sys, pset, autexpos, k, autconv)

        next!(p)
    end
    print("\nAutonomous parametrization complete.\n")
end
