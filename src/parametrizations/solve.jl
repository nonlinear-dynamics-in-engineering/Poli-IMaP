# Solve the homological equations given a list of homogeneous monomials.
# Store the result back into WPoly and fPoly
function solve_homological_equations!(sys::System,
    pset::ParSettings,
    expos::Vector,
    E::SparseArray{ComplexF64};
    W₁::Any = nothing,
    addToCurrentW::Bool = false,
    addToCurrentf::Bool = false)

    nMonomials = length(expos)

    if pset.fullspectrum

        # solve system in modal coordinates
        η = - sys.l_eigvecs * E
        ξ = SparseArray(zeros(ComplexF64, sys.ddim, nMonomials))
        ϕ = SparseArray(zeros(ComplexF64, sys.rdim, nMonomials))

        if pset.parstyle == "normal-form"
            for (expoidx, expo) in enumerate(expos)
                eigSum = dot(expo .- 1, sys.eigvals[pset.tgaxes])
                for normidx in pset.normaxes
                    if abs(eigSum - sys.eigvals[normidx]) > pset.cross_res_tol
                        ξ[normidx, expoidx] = η[normidx, expoidx]/(sys.eigvals[normidx] - eigSum)
                    else
                        error("Found cross resonance condition below tolerance.
                        Index: $normidx, Exponents: $expo")
                    end
                end

                for tgidx in pset.auttgaxes
                    if abs(eigSum - sys.eigvals[tgidx]) > pset.internal_res_tol
                        ξ[tgidx, expoidx] = η[tgidx, expoidx]/(sys.eigvals[tgidx] - eigSum)
                    else
                        error("Found internal resonance condition below tolerance.
                        Index: $tgidx, Exponents: $expo")
                    end
                end
            end

        elseif pset.parstyle == "graph"
            if isnothing(W₁)
                for (expoidx, expo) in enumerate(expos)
                    for normidx in pset.normaxes
                        if abs(eigSum - sys.eigvals[normidx]) > pset.cross_res_tol
                            ξ[normidx, expoidx] = η[normidx, expoidx]/(sys.eigvals[normidx] - eigSum)
                        else
                            error("Found cross resonance condition below tolerance.
                            Index: $normidx, Exponents: $expo")
                        end
                    end

                    for (rdidx, tgidx) in zip(pset.autrdaxes, pset.auttgaxes)
                        ϕ[rdidx, expoidx] = -η[tgidx, expoidx]
                    end
                end
            else
                @goto solve_full_system
            end

        elseif pset.parstyle == "resonant"
            if isnothing(W₁)
                for (expoidx, expo) in enumerate(expos)
                    eigSum = dot(expo .- 1, sys.eigvals[pset.tgaxes])
                    resonantSet = Int[]
                    for eigidx in pset.autrdaxes
                        if expo in pset.resexpos[eigidx]
                            push!(resonantSet, eigidx)
                        end
                    end

                    for normidx in pset.normaxes
                        if abs(eigSum - sys.eigvals[normidx]) > pset.cross_res_tol
                            ξ[normidx, expoidx] = η[normidx, expoidx]/(sys.eigvals[normidx] - eigSum)
                        else
                            error("Found cross resonance condition below tolerance.
                            Index: $normidx, Exponents: $expo")
                        end
                    end

                    for (rdidx, tgidx) in zip(pset.autrdaxes, pset.auttgaxes)
                        if rdidx ∉ resonantSet
                            if abs(eigSum - sys.eigvals[tgidx]) > pset.internal_res_tol
                                ξ[tgidx, expoidx] = η[tgidx, expoidx]/(sys.eigvals[tgidx] - eigSum)
                            else
                                error("Found internal resonance condition below tolerance.
                                Index: $tgidx, Exponents: $expo")
                            end
                        else
                            ϕ[rdidx, expoidx] = -η[tgidx, expoidx]
                        end
                    end
                end
            else
                @goto solve_full_system
            end
        end

        # add newly found Wₖ, fₖ and DWₖ to polynomial format
        update_poly!(sys.W, sys.r_eigvecs * ξ, expos, addToCurrent=addToCurrentW)
        update_poly!(sys.f, ϕ, expos, addToCurrent=addToCurrentf)

    else
        # solve system in physical coordinates
        @label solve_full_system

        Wₖ    = SparseArray(zeros(ComplexF64, sys.ddim, nMonomials))
        fₖ    = SparseArray(zeros(ComplexF64, sys.rdim, nMonomials))
        axes  = vcat(pset.normaxes, pset.auttgaxes)
        nAxes = length(axes)
        nrdaxes = length(pset.autrdaxes)

        if isnothing(W₁)
            Yₗ = sys.r_eigvecs[:, pset.auttgaxes]
            Xₗ = sys.l_eigvecs[pset.auttgaxes, :]
        else
            Yₗ = W₁[:, pset.autrdaxes]
            Xₗ = sys.l_eigvecs[pset.auttgaxes, :]
        end

        for (expoidx, expo) in enumerate(expos)
            eigSum = dot(expo .- 1, sys.eigvals[pset.tgaxes])
            RHS    = Array(E[axes, expoidx])
            LHS    = (eigSum * sys.B₀ - sys.jacobian)[axes, axes]

            if pset.parstyle == "normal-form"
                for eigidx in pset.auttgaxes
                    if abs(eigSum - sys.eigvals[eigidx]) < pset.internal_res_tol
                        error("Found internal resonance condition below tolerance.
                        Index: $eigidx, Exponents: $expo")
                    end
                end
                Wₖ[axes, expoidx] = LHS \ RHS

            elseif pset.parstyle == "resonant"
                resonantSet = Int[]
                for eigidx in pset.autrdaxes
                    if expo in pset.resexpos[eigidx]
                       push!(resonantSet, eigidx)
                    end
                end
                nres = length(resonantSet)
                if nres > 0
                    bigLHS = SparseArray(zeros(ComplexF64, nAxes + nrdaxes, nAxes + nrdaxes))
                    bigLHS[1:nAxes, 1:nAxes] = LHS
                    bigLHS[1:nAxes, nAxes + 1:nAxes + nres] = sys.B₀[axes, axes] * Yₗ[axes, resonantSet]
                    bigLHS[nAxes + 1:nAxes + nres, 1:nAxes] = Xₗ[resonantSet, axes] * sys.B₀[axes, axes]
                    bigLHS[nAxes + nres + 1:end, nAxes + nres + 1:end] = I(nrdaxes - nres)
                    bigRHS = SparseArray(zeros(ComplexF64, nAxes + nrdaxes))
                    bigRHS[1:nAxes] = RHS
                    bigSolution = Array(bigLHS) \ Array(bigRHS)
                    Wₖ[axes, expoidx] = bigSolution[1:nAxes]
                    fₖ[resonantSet, expoidx] = bigSolution[nAxes + 1:nAxes + nres]
                else
                    Wₖ[axes, expoidx] = LHS \ RHS
                end

            elseif pset.parstyle == "graph"
                bigLHS = SparseArray(zeros(ComplexF64, nAxes + nrdaxes, nAxes + nrdaxes))
                bigLHS[1:nAxes, 1:nAxes] = LHS
                bigLHS[1:nAxes, nAxes + 1:nAxes + nrdaxes] = sys.B₀[axes, axes] * Yₗ[axes, :]
                bigLHS[nAxes + 1:end, 1:nAxes] = Xₗ[:, axes] * sys.B₀[axes, axes]
                bigRHS = SparseArray(zeros(ComplexF64, nAxes + nrdaxes))
                bigRHS[1:nAxes] = RHS
                bigSolution = Array(bigLHS) \ Array(bigRHS)
                Wₖ[axes, expoidx] = bigSolution[1:nAxes]
                fₖ[pset.autrdaxes, expoidx] = bigSolution[nAxes + 1:end]
            else
                error("Parametrization style not properly specified")
            end
        end

        update_poly!(sys.W, Wₖ, expos, addToCurrent=addToCurrentW)
        update_poly!(sys.f, fₖ, expos, addToCurrent=addToCurrentf)
    end
end
