
function _chebyshev_nodes(a::Real, b::Real, n::Int)
    nodes = [(a + b)/2 + (b - a)/2 * cospi((2k - 1) / (2n)) for k in 1:n]
    return sort!(nodes)
end

_eval_entry(x::Number,        _)     = ComplexF64(x)

function _eval_entry(x::Num,  subst)
    v = Symbolics.value(Symbolics.substitute(x, subst))
    return ComplexF64(v)
end

function _eval_entry(x::Complex{Num}, subst)
    r = Symbolics.value(Symbolics.substitute(real(x), subst))
    i = Symbolics.value(Symbolics.substitute(imag(x), subst))
    return complex(Float64(r), Float64(i))
end

function _eval_matrix_at(A::Matrix, param::Num, val::Real)
    subst = Dict(param => val)
    return [_eval_entry(A[i, j], subst) for i in 1:size(A,1), j in 1:size(A,2)]
end

function _jacobian_has_symbols(J::Matrix)
    for entry in J
        vars = _entry_variables(entry)
        isempty(vars) || return true
    end
    return false
end

_entry_variables(x::Num)          = Symbolics.get_variables(x)
_entry_variables(x::Complex{Num}) = union(Symbolics.get_variables(real(x)),
                                          Symbolics.get_variables(imag(x)))
_entry_variables(::Number)        = []


function _match_eigenvalues(prev::Vector{ComplexF64}, curr::Vector{ComplexF64})
    n    = length(prev)
    perm = zeros(Int, n)
    used = falses(n)
    for k in 1:n
        best_j    = 0
        best_dist = Inf
        for j in 1:n
            used[j] && continue
            d = abs(prev[k] - curr[j])
            if d < best_dist
                best_dist = d
                best_j    = j
            end
        end
        perm[k]      = best_j
        used[best_j] = true
    end
    return perm
end

function _chebyshev_basis(t::Float64, degree::Int)
    T    = Vector{Float64}(undef, degree + 1)
    T[1] = 1.0
    degree == 0 && return T
    T[2] = t
    for j in 3:degree+1
        T[j] = 2t * T[j-1] - T[j-2]
    end
    return T
end

function _fit_cheb_coeffs(xs::Vector{Float64}, ys::Vector{ComplexF64}, degree::Int)
    a, b = xs[1], xs[end]
    ts   = @. 2 * (xs - a) / (b - a) - 1

    deg1 = degree + 1
    V    = zeros(length(xs), deg1)
    for (i, t) in enumerate(ts)
        V[i, :] = _chebyshev_basis(t, degree)
    end

    c_real = V \ real.(ys)
    c_imag = V \ imag.(ys)
    return a, b, c_real, c_imag
end

function _cheb_to_monomial_coeffs(a::Float64, b::Float64, c::Vector{Float64})
    d = length(c) - 1

    # Build T_k(τ) as polynomial coefficient vectors in τ using the recurrence
    # T_0 = 1, T_1 = τ, T_k = 2τ T_{k-1} - T_{k-2}
    T = Vector{Vector{Float64}}(undef, d + 1)
    T[1] = [1.0]
    d >= 1 && (T[2] = [0.0, 1.0])
    for k in 3:d+1
        prev   = T[k-1]
        shifted = vcat(0.0, 2.0 .* prev)                            # 2τ * T_{k-1}
        prev2  = vcat(T[k-2], zeros(length(shifted) - length(T[k-2])))
        T[k]   = shifted .- prev2
    end

    # Sum c[k] * T[k] to get monomial coefficients in τ
    p_τ = zeros(d + 1)
    for k in 1:d+1, j in 1:length(T[k])
        p_τ[j] += c[k] * T[k][j]
    end

    # Substitute τ = α*x + β (α = 2/(b-a), β = -(a+b)/(b-a)) to get coefficients in x
    α = 2.0 / (b - a)
    β = -(a + b) / (b - a)
    p_x   = zeros(d + 1)
    power = [1.0]                 # (α*x + β)^0 = 1
    for k in 0:d
        for j in eachindex(power)
            p_x[j] += p_τ[k+1] * power[j]
        end
        # Next power: (α*x + β)^(k+1) = (β + α*x) * current
        new_power = zeros(length(power) + 1)
        new_power[1:length(power)]   .+= β .* power
        new_power[2:length(power)+1] .+= α .* power
        power = new_power
    end

    return p_x
end

function _cheb_to_symbolic(param::Num, a::Float64, b::Float64,
                            c_real::Vector{Float64}, c_imag::Vector{Float64})
    p_real = _cheb_to_monomial_coeffs(a, b, c_real)
    p_imag = _cheb_to_monomial_coeffs(a, b, c_imag)

    function mono_sum(p::Vector{Float64})
        expr = Num(0)
        for (k, coeff) in enumerate(p)
            iszero(coeff) && continue
            expr += k == 1 ? coeff : coeff * param^(k - 1)
        end
        return expr
    end

    return Complex{Num}(mono_sum(p_real), mono_sum(p_imag))
end

function _eval_matrix_at(A::Matrix, params::Vector{Num}, vals::Vector{Float64})
    subst = Dict(params[i] => vals[i] for i in eachindex(params))
    return [_eval_entry(A[i, j], subst) for i in 1:size(A,1), j in 1:size(A,2)]
end

function _chebyshev_grid(intervals::Vector{Tuple{Float64,Float64}}, npoints::Int)
    nodes_per_dim = [_chebyshev_nodes(a, b, npoints) for (a, b) in intervals]
    grid = Vector{Vector{Float64}}()
    _build_grid!(grid, nodes_per_dim, Float64[], 1)
    return grid
end

function _build_grid!(grid, nodes_per_dim, partial, dim)
    if dim > length(nodes_per_dim)
        push!(grid, copy(partial))
        return
    end
    for x in nodes_per_dim[dim]
        push!(partial, x)
        _build_grid!(grid, nodes_per_dim, partial, dim + 1)
        pop!(partial)
    end
end

function _monomial_exponents(nvars::Int, degree::Int)
    exponents = Vector{Vector{Int}}()
    _enum_exponents!(exponents, Int[], nvars, degree)
    sort!(exponents, by = e -> (sum(e), e))
    return exponents
end

function _enum_exponents!(out, partial, remaining_vars, remaining_deg)
    if remaining_vars == 0
        push!(out, copy(partial))
        return
    end
    for k in 0:remaining_deg
        push!(partial, k)
        _enum_exponents!(out, partial, remaining_vars - 1, remaining_deg - k)
        pop!(partial)
    end
end

function _multivar_vandermonde(grid::Vector{Vector{Float64}},
                               exponents::Vector{Vector{Int}})
    npts = length(grid)
    nmon = length(exponents)
    V    = zeros(npts, nmon)
    for i in 1:npts, j in 1:nmon
        v = 1.0
        for k in eachindex(exponents[j])
            exponents[j][k] > 0 && (v *= grid[i][k]^exponents[j][k])
        end
        V[i, j] = v
    end
    return V
end

function _multivar_to_symbolic(params::Vector{Num},
                                exponents::Vector{Vector{Int}},
                                c_real::Vector{Float64},
                                c_imag::Vector{Float64})
    function mono_sum(coeffs::Vector{Float64})
        expr = Num(0)
        for (j, coeff) in enumerate(coeffs)
            iszero(coeff) && continue
            term = coeff
            for (p, k) in zip(params, exponents[j])
                k > 0 && (term = term * p^k)
            end
            expr += term
        end
        return expr
    end
    return Complex{Num}(mono_sum(c_real), mono_sum(c_imag))
end

function interpolate_spectrum(A::Matrix,
                              param::Num,
                              interval::Tuple{<:Real,<:Real};
                              npoints::Int = 40,
                              degree::Int  = npoints - 1)

    a, b = Float64(interval[1]), Float64(interval[2])
    @assert a < b       "interval must satisfy a < b"
    @assert npoints ≥ 2 "npoints must be ≥ 2"
    degree = min(degree, npoints - 1)

    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"

    # 1. Sample A at Chebyshev nodes
    xs = _chebyshev_nodes(a, b, npoints)   # sorted ascending

    raw_eigvals = Vector{Vector{ComplexF64}}(undef, npoints)
    raw_eigvecs = Vector{Matrix{ComplexF64}}(undef, npoints)   # un-normalised

    for (i, x) in enumerate(xs)
        Anum = _eval_matrix_at(A, param, x)
        eig  = eigen(Anum)
        perm = sortperm(eig.values, by = v -> (real(v), imag(v)))
        raw_eigvals[i] = eig.values[perm]
        raw_eigvecs[i] = eig.vectors[:, perm]
    end

    # 2. Track eigenvalue branches for continuity
    #    tracked_evals[k][i]        = eigenvalue k at sample i
    #    tracked_revecs[row][k][i]  = component row of right eigvec k at i
    #    tracked_levecs[k][row][i]  = component row of left  eigvec k at i
    tracked_evals  = [Vector{ComplexF64}(undef, npoints) for _ in 1:n]
    tracked_revecs = [[Vector{ComplexF64}(undef, npoints) for _ in 1:n] for _ in 1:n]

    for k in 1:n
        tracked_evals[k][1] = raw_eigvals[1][k]
        for row in 1:n
            tracked_revecs[row][k][1] = raw_eigvecs[1][row, k]
        end
    end

    for i in 2:npoints
        prev = [tracked_evals[k][i-1] for k in 1:n]
        perm = _match_eigenvalues(prev, raw_eigvals[i])
        for k in 1:n
            tracked_evals[k][i] = raw_eigvals[i][perm[k]]
            for row in 1:n
                tracked_revecs[row][k][i] = raw_eigvecs[i][row, perm[k]]
            end
        end
    end

    # Left eigenvectors: inv(V) at each sample, where V is the tracked
    # (reordered) right-eigenvector matrix.  Rows of inv(V) are left eigvecs.
    tracked_levecs = [[Vector{ComplexF64}(undef, npoints) for _ in 1:n] for _ in 1:n]

    for i in 1:npoints
        V = [tracked_revecs[row][k][i] for row in 1:n, k in 1:n]
        L = inv(V)   # L[k, row] = component row of left eigvec k
        for k in 1:n, row in 1:n
            tracked_levecs[k][row][i] = L[k, row]
        end
    end

    # 3. Fit Chebyshev polynomials → symbolic expressions
    xs_f64 = Vector{Float64}(xs)

    eigval_exprs = Vector{Complex{Num}}(undef, n)
    for k in 1:n
        a0, b0, cr, ci = _fit_cheb_coeffs(xs_f64, tracked_evals[k], degree)
        eigval_exprs[k] = _cheb_to_symbolic(param, a0, b0, cr, ci)
    end

    r_eigvec_exprs = Matrix{Complex{Num}}(undef, n, n)
    for k in 1:n, row in 1:n
        a0, b0, cr, ci = _fit_cheb_coeffs(xs_f64, tracked_revecs[row][k], degree)
        r_eigvec_exprs[row, k] = _cheb_to_symbolic(param, a0, b0, cr, ci)
    end

    l_eigvec_exprs = Matrix{Complex{Num}}(undef, n, n)
    for k in 1:n, row in 1:n
        a0, b0, cr, ci = _fit_cheb_coeffs(xs_f64, tracked_levecs[k][row], degree)
        l_eigvec_exprs[k, row] = _cheb_to_symbolic(param, a0, b0, cr, ci)
    end

    return eigval_exprs, r_eigvec_exprs, l_eigvec_exprs
end

function interpolate_spectrum(A::Matrix,
                              params::Vector{Num},
                              intervals::Vector{<:Tuple{<:Real,<:Real}};
                              npoints::Int = 10,
                              degree::Int  = 3)

    m = length(params)
    @assert length(intervals) == m "Need one interval per parameter (got $(length(intervals)) for $m parameters)"
    @assert all(t -> t[1] < t[2], intervals) "Each interval must satisfy a < b"
    @assert npoints ≥ 2 "npoints must be ≥ 2"

    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"

    intervals_f64 = [(Float64(a), Float64(b)) for (a, b) in intervals]

    # 1. Build grid and monomial basis
    grid      = _chebyshev_grid(intervals_f64, npoints)
    exponents = _monomial_exponents(m, degree)
    ngrid     = length(grid)
    nmon      = length(exponents)
    @assert ngrid ≥ nmon "Need at least $nmon grid points for degree-$degree " *
                         "in $m variables, but npoints=$npoints gives only $ngrid. " *
                         "Increase npoints."

    V_vander = _multivar_vandermonde(grid, exponents)

    # 2. Evaluate eigen-pairs at every grid point
    raw_eigvals = Vector{Vector{ComplexF64}}(undef, ngrid)
    raw_eigvecs = Vector{Matrix{ComplexF64}}(undef, ngrid)

    for (i, pt) in enumerate(grid)
        Anum = _eval_matrix_at(A, params, pt)
        eig  = eigen(Anum)
        perm = sortperm(eig.values, by = v -> (real(v), imag(v)))
        raw_eigvals[i] = eig.values[perm]
        raw_eigvecs[i] = eig.vectors[:, perm]
    end

    # 3. Track eigenvalue branches across the grid
    tracked_evals  = [Vector{ComplexF64}(undef, ngrid) for _ in 1:n]
    tracked_revecs = [[Vector{ComplexF64}(undef, ngrid) for _ in 1:n] for _ in 1:n]

    for k in 1:n
        tracked_evals[k][1] = raw_eigvals[1][k]
        for row in 1:n
            tracked_revecs[row][k][1] = raw_eigvecs[1][row, k]
        end
    end

    for i in 2:ngrid
        prev = [tracked_evals[k][i-1] for k in 1:n]
        perm = _match_eigenvalues(prev, raw_eigvals[i])
        for k in 1:n
            tracked_evals[k][i] = raw_eigvals[i][perm[k]]
            for row in 1:n
                tracked_revecs[row][k][i] = raw_eigvecs[i][row, perm[k]]
            end
        end
    end

    # Left eigenvectors
    tracked_levecs = [[Vector{ComplexF64}(undef, ngrid) for _ in 1:n] for _ in 1:n]
    for i in 1:ngrid
        R = [tracked_revecs[row][k][i] for row in 1:n, k in 1:n]
        L = inv(R)
        for k in 1:n, row in 1:n
            tracked_levecs[k][row][i] = L[k, row]
        end
    end

    # 4. Least-squares polynomial fit → symbolic expressions
    eigval_exprs = Vector{Complex{Num}}(undef, n)
    for k in 1:n
        cr = V_vander \ real.(tracked_evals[k])
        ci = V_vander \ imag.(tracked_evals[k])
        eigval_exprs[k] = _multivar_to_symbolic(params, exponents, cr, ci)
    end

    r_eigvec_exprs = Matrix{Complex{Num}}(undef, n, n)
    for k in 1:n, row in 1:n
        cr = V_vander \ real.(tracked_revecs[row][k])
        ci = V_vander \ imag.(tracked_revecs[row][k])
        r_eigvec_exprs[row, k] = _multivar_to_symbolic(params, exponents, cr, ci)
    end

    l_eigvec_exprs = Matrix{Complex{Num}}(undef, n, n)
    for k in 1:n, row in 1:n
        cr = V_vander \ real.(tracked_levecs[k][row])
        ci = V_vander \ imag.(tracked_levecs[k][row])
        l_eigvec_exprs[k, row] = _multivar_to_symbolic(params, exponents, cr, ci)
    end

    return eigval_exprs, r_eigvec_exprs, l_eigvec_exprs
end
