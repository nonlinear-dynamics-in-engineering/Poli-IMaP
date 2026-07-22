using Poli_IMAP

# the ziegler column (a follower-force double pendulum) under a pulsating
# follower load. states x1..x4 are the two mechanical dofs; x5,x6 are a harmonic
# block rotating at Ω that carries the time-periodic forcing. we build the 4-d
# invariant manifold — autonomous master modes 1,2 plus nonautonomous forcing
# modes 5,6 — in resonant style: first the autonomous part, then the
# nonautonomous extension.

const D = Complex{Float64}
# variables μ, ξ₁
const μ = 0.4408
const ξ₁ = 0.1
const ξ₂ = 0.1
const Δμ = 1.0
const Ω = 2 * 1.6906
const ddim = 6
const rdim = 4
const randim = 6

# the ziegler column system
function F(x)
    return [x[2],
            (cos(x[1] - x[3]) * (-1.0 * x[3] + x[1] - ξ₂ * x[4] + ξ₂ * x[2])
            + sin(x[1] - x[3]) * (x[2] * x[2] * cos(x[1] - x[3]) + x[4] * x[4] - μ - Δμ * x[5])
            + 2 * x[1] - x[3] + (ξ₁ + ξ₂) * x[2] - ξ₂ * x[4])/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3),

            x[4],

            (sin(x[1] - x[3]) * (-1.0 * x[4] * x[4] * cos(x[1] - x[3]) +
            (μ + Δμ * x[5]) * cos(x[1] - x[3]) - 3 * x[2] * x[2])
            + cos(x[1] - x[3]) * (- 2 * x[1] + x[3] - (ξ₁ + ξ₂) * x[2] + ξ₂ * x[4])
            + 3 * x[3] - 3.0 * x[1] + 3 * ξ₂ * (x[4] - x[2]))/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3),

            - Ω * x[6],

            Ω * x[5]]
end

sys = System{D}(F=F, ddim=6, rdim=4, D=D, maxorder=20);

initialize_system!(sys);
linearize_system!(sys);

settings() = (autidxs = [1, 2, 3, 4],
    nonautidxs = [5, 6],
    tgaxes     = [1, 2, 5, 6],
    auttgaxes  = [1, 2],
    nonauttgaxes = [5, 6],
    fullspectrum = true,
    parstyle   = "resonant",
    resonances = Dict(1 => (1, -1, 2, -2), 2 => (-1, 1, -2, 2)),
    kaut       = 10,
    knonaut    = 5);

pset = set_parametrization_settings(sys; settings()...);

parametrize_autonomous!(sys, pset);

parametrize_nonautonomous!(sys, pset);
