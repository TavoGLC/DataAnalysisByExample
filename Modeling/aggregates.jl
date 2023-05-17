
###############################################################################
# Loading packages 
###############################################################################

using DifferentialEquations
using Plots

###############################################################################
# Loading packages 
###############################################################################

function Polymer01!(du,u,p,t)
    
    u1,u2 = u
    n,k1pn,k2pn = p
    
    du[1] = -k1pn*u1^(n+1) - k2pn*u1*u2
    du[2] = k1pn*u1^(n+1)

    nothing
end

u0s = [50,0]
params = [1,1.4*10^-4,5*10^-3]

prob = ODEProblem(Polymer01!,u0s,(0.0,100),params)
sol = solve(prob)
plot(sol,tspan=(0,100))

###############################################################################
# Loading packages 
###############################################################################
function Polymer02!(du,u,p,t)
    
    u1,u2 = u
    k1pn,k2pn = p
    
    du[1] = -k1pn*u1 - k2pn*u1*u2
    du[2] = k1pn*u1 + k2pn*u1*u2

    nothing
end

u0s = [50,0]
params = [2.5*10^-3,3.7*10^-3]

prob = ODEProblem(Polymer02!,u0s,(0.0,200),params)
sol = solve(prob)
plot(sol,tspan=(0,200))

###############################################################################
# Loading packages 
###############################################################################

function Polymer03!(du,u,p,t)
    
    u1,u2,u3 = u
    n,k1sg,k2sg,k3sg = p
    
    du[1] = -n*k1sg*u1^n - k3sg*u1*u3
    du[2] = k1sg*u1^n - 2*k2sg*u2
    du[3] = k2sg*u2

    nothing
end

u0s = [50,0,0]
params = [1,1.4*10^-1,2.9*10^-3,1.3*10^-1]

prob = ODEProblem(Polymer03!,u0s,(0.0,200),params)
sol = solve(prob)
plot(sol,tspan=(0,200))
