include("../import_all.jl")

# high-dimension test (ddim = 1000). F is a
# random nonlinear field with sparse entries; the example parametrizes a small invariant
# manifold (autonomous master modes 1,2 plus nonautonomous forcing 5,6) to
# check how it scales  with ddim. F uses rand(), so the system
# differs run to run.

const D = Complex{Float64}
const a = 1.0
const b = 0.5
const c = 0.2
const ddim = 1000
const rdim = 4

function F(x)
    return [(-rand(1)[1]*x[i] 
    -x[rand(1:ddim)] 
    -(b/10)*x[i]*x[rand(1:ddim)] 
    - (c/10)*x[i]*x[rand(1:ddim)]*x[i]
    +rand(1)[1]*sin(x[i]-x[rand(1:ddim)])*cos(3*x[i])) 
    for i in 1:ddim]
end

sys = System{D}(F=F, ddim=ddim, rdim=rdim, D=D, maxorder=20);

initialize_system!(sys);
linearize_system!(sys, compute_spectrum=true);

pset = set_parametrization_settings(sys,    
    autidxs    = union([1, 2, 3, 4], collect(7:ddim)),
    nonautidxs = [5, 6],
    tgaxes     = [1, 2, 5, 6],
    auttgaxes  = [1, 2],
    nonauttgaxes = [5, 6],
    fullspectrum = true,
    parstyle   = "resonant",
    resonances = Dict(1 => (1, -1, 2, -2),
                      2 => (-1, 1, -2, 2)),
    kaut       = 5,
    knonaut    = 5,
);

parametrize_autonomous!(sys, pset);
parametrize_nonautonomous!(sys, pset);