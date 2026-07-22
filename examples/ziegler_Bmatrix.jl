using Poli_IMAP

# the ziegler column with a pulsating follower force, written in its natural form
# B(x)·ẋ = F(x) with the mass matrix B(x) kept explicit instead of baking B⁻¹
# into the field (see the figure). states x = [θ1, θ2, θ̇1, θ̇2, x5, x6]: the four
# mechanical states plus the x5,x6 harmonic block (frequency Ω) carrying the
# pulsating load. same parameters and parametrization settings as ziegler.jl;
# only the B handling differs (so the higher-order terms differ but the linear
# part, spectrum and settings match).

const D = Complex{Float64}
const μ  = 0.4408
const ξ₁ = 0.1
const ξ₂ = 0.1
const Δμ = 1.0
const Ω  = 2 * 1.6906
const ddim = 6
const rdim = 4
const randim = 6

function F(x)
    sn = sin(x[1] - x[2])
    return [x[3],
            x[4],
            -x[4]^2 * sn - 2*x[1] + x[2] - (ξ₁ + ξ₂)*x[3] + ξ₂*x[4] + (μ + Δμ*x[5])*sn,
             x[3]^2 * sn -   x[2] +   x[1] - ξ₂*(x[4] - x[3]),
            -Ω * x[6],
             Ω * x[5]]
end

function B(x)
    c = cos(x[1] - x[2])
    return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # col 1
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,    # col 2
            0.0, 0.0, 3.0, c,   0.0, 0.0,    # col 3
            0.0, 0.0, c,   1.0, 0.0, 0.0,    # col 4
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,    # col 5
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0]    # col 6
end

sys = System{D}(F=F, B=B, ddim=ddim, rdim=rdim, D=D, maxorder=20);

initialize_system!(sys);
linearize_system!(sys);

pset = set_parametrization_settings(sys;
    autidxs    = [1, 2, 3, 4],
    nonautidxs = [5, 6],
    tgaxes     = [1, 2, 5, 6],
    auttgaxes  = [1, 2],
    nonauttgaxes = [5, 6],
    fullspectrum = true,
    parstyle   = "resonant",
    resonances = Dict(1 => (1, -1, 2, -2),
                      2 => (-1, 1, -2, 2)),
    kaut       = 12,
    knonaut    = 5,
);

parametrize_autonomous!(sys, pset);

parametrize_nonautonomous!(sys, pset);
