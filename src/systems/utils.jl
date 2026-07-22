# Initialize a system by building a dag
function initialize_system!(sys::System{D}) where D

    print("Building graph of F... ")
    sys.gF, sys.gFn = record_pgraph(
        sys.F, sys.ddim, sys.rdim, sys.ddim, sys.maxorder, D)
    println("done")

    if sys.B isa Function
        sys.B_is_const = false

        # Row map: row r of the B-series ↔ matrix entry B[i, j].
        #   dense  (Bpattern === nothing): all ddim² entries, column-major.
        #   sparse (Bpattern given):       only the listed nonzero entries.
        if isnothing(sys.Bpattern)
            sys.B_sparse = false
            sys.B_rowmap = [(i, j) for j in 1:sys.ddim for i in 1:sys.ddim]
        else
            sys.B_sparse = true
            sys.B_rowmap = sys.Bpattern
        end
        nB = length(sys.B_rowmap)

        print("Building graph of B... ")
        sys.gB, sys.gBn = record_pgraph(
            sys.B, sys.ddim, sys.rdim, nB, sys.maxorder, D)
        println(sys.B_sparse ? "done ($nB nonzero entries)." : "done.")

        print("Computing B(0)... ")
        # B(0) from the *numeric* eval tape gBn (already swept at x=0 in
        # record_pgraph). gBn.data.y is the flat length-nB vector of entries; we
        # scatter it into the ddim×ddim B₀ via the row map. invB₀ is computed in
        # linearize_system!.
        sys.flatB₀ = copy(sys.gBn.data.y)
        sys.B₀     = zeros(D, sys.ddim, sys.ddim)
        @inbounds for (r, (i, j)) in enumerate(sys.B_rowmap)
            sys.B₀[i, j] = sys.flatB₀[r]
        end
        println("done\n")

        sys.BDWf = PolynomialArray{sys.D}(sys.maxorder, sys.rdim, sys.ddim)

    else
        print("Building graph of B... ")
        println("B matrix found to be constant... done")
        sys.B_is_const = true
        if isnothing(sys.B)
            sys.B   = Matrix{D}(I, sys.ddim, sys.ddim)
            sys.B₀  = Matrix{D}(I, sys.ddim, sys.ddim)
            sys.invB₀ = Matrix{D}(I, sys.ddim, sys.ddim)
        else
            sys.B₀    = sys.B
            sys.invB₀ = inv(sys.B)
        end
    end

    sys.W   = PolynomialArray{sys.D}(sys.maxorder, sys.rdim, sys.ddim)
    sys.f   = PolynomialArray{sys.D}(sys.maxorder, sys.rdim, sys.rdim)
    sys.DW  = PolynomialArray{sys.D}(sys.maxorder, sys.rdim, sys.ddim * sys.rdim)
    sys.DWk = PolynomialArray{sys.D}(sys.maxorder, sys.rdim, sys.ddim * sys.rdim)
    sys.DWf = PolynomialArray{sys.D}(sys.maxorder, sys.rdim, sys.ddim)

    sys.initialized = true
    return true
end


# Find the linearized system
function linearize_system!(sys::System;
            compute_spectrum::Bool = true,
            param::Union{Nothing, Num, Vector{Num}} = nothing,
            interval::Union{Nothing, Tuple{<:Real,<:Real}, Vector{<:Tuple{<:Real,<:Real}}} = nothing,
            npoints::Int = -1,
            degree::Int = -1)

    if sys.initialized == false
        error("System not initialized!")
    end

    print("Computing Jacobian of F... ")

    # Compute J via reverse-mode AD: one scalar sweep per output row.
    # O(ddim × nnodes) complexity
    # fwd_evaluation_sweep! is called once; rev_adjoint_sweep! is called ddim times.
    x0 = zeros(sys.D, sys.ddim)
    fwd_evaluation_sweep!(sys.gFn, x0)
    J  = zeros(sys.D, sys.ddim, sys.ddim)
    yBar = zeros(sys.D, sys.ddim)
    for i in 1:sys.ddim
        yBar[i] = one(sys.D)
        rev_adjoint_sweep!(sys.gFn, yBar)
        @inbounds for j in 1:sys.ddim
            J[i, j] = sys.gFn.data.xBar[j]
        end
        yBar[i] = zero(sys.D)
    end

    sys.jacobian = J
    sys.invB₀    = inv(sys.B₀)
    # Linearized operator of ẋ = B(x)⁻¹F(x) is B₀⁻¹J. For constant B = I this is
    # just J; for a non-constant B it folds in the mass matrix, matching the
    # invB₀ factor already used in solve_homological_equations!
    linJ = sys.invB₀ * sys.jacobian
    println("done")

    if compute_spectrum
        if _jacobian_has_symbols(linJ)
            # Symbolic Jacobian: interpolate spectrum over the parameter interval
            if isnothing(param) || isnothing(interval)
                @warn "Jacobian contains symbolic parameters but no `param`/`interval` " *
                      "were provided to linearize_system!. Skipping spectrum computation."
            elseif param isa Num
                # Single parameter
                npts = npoints < 0 ? 40 : npoints
                deg  = degree  < 0 ? npts - 1 : degree
                print("Computing full spectrum by interpolation (1 parameter)... ")
                sys.eigvals, sys.r_eigvecs, sys.l_eigvecs =
                    interpolate_spectrum(linJ, param, interval::Tuple;
                                         npoints = npts, degree = deg)
                sys.l_eigvecs = sys.l_eigvecs * sys.invB₀   # B₀-orthonormal
                println("done")
            else
                # Multiple parameters
                npts = npoints < 0 ? 10 : npoints
                deg  = degree  < 0 ? 3  : degree
                print("Computing full spectrum by interpolation ($(length(param)) parameters)... ")
                sys.eigvals, sys.r_eigvecs, sys.l_eigvecs =
                    interpolate_spectrum(linJ, param, interval::Vector;
                                         npoints = npts, degree = deg)
                sys.l_eigvecs = sys.l_eigvecs * sys.invB₀   # B₀-orthonormal
                println("done")
            end
        else
            # Purely numeric Jacobian: direct eigendecomposition
            print("Computing full spectrum... ")
            eig  = eigen(linJ)
            perm = sortperm(eig.values, by = x -> (round(real(x), digits=10), imag(x)))
            sys.eigvals   = eig.values[perm]
            sys.r_eigvecs = eig.vectors[:, perm]
            # generalized (B₀-orthonormal) left eigenvectors: l·B₀·r = I
            sys.l_eigvecs = inv(sys.r_eigvecs) * sys.invB₀
            println("done")
        end
    end
    return true
end


# populate gF/gFn node partials (SparsePartials) for use by update_sweep!
# called once at the start of parametrize_autonomous! — not during linearize_system!
function _compute_jacobian!(sys::System)
    fwd_adjoint_sweep!(sys.gFn, zeros(sys.D, sys.ddim))
    for ndidx in 1:length(sys.gFn.ndlist)
        sys.gF.ndlist[ndidx].data.partials           = sys.gFn.ndlist[ndidx].data.partials
        sys.gF.ndlist[ndidx].data.extradata.partials = sys.gFn.ndlist[ndidx].data.extradata.partials
    end
end
