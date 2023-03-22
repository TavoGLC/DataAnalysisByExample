"""
MIT License
Copyright (c) 2023 Octavio Gonzalez-Lugo 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: Octavio Gonzalez-Lugo

Model 
An efective model of endogenous clocks and external stimuli determining circadian rhythms
Tim Breitenbach, Charlotte Helfrich‑Förster & Thomas Dandekar

"""

###############################################################################
# Loading packages 
###############################################################################

using DifferentialEquations
using Plots

###############################################################################
# Loading packages 
###############################################################################

function f(Vm,Km,V)
    Vm*(V/(V+Km))
end

function f0(Vm,Km,KI,n,V)
    Vm*((Km^n)/((V^n)+(KI^n)))
end

function Goldbeter!(du,u,p,t)
    
    u1,u2,u3,u4,u5= u
    n,KI,Km,K1,K2,K3,K4,Kd,vs,vm,ks,V1,V2,V3,V4,k1,k2,vd = p
    
    du[1] = f0(vs,K1,KI,n,u5) - f(vm,Km,u1)
    du[2] = ks*u1 - f(V1,K1,u2) + f(V2,K2,u3)
    du[3] =  f(V1,K1,u1) - f(V2,K2,u3) - f(V3,K3,u3) + f(V4,K4,u4)
    du[4] =  f(V3,K3,u3) - f(V4,K4,u4) - k1*u4 + k2*u5 - f(vd,Kd,u4)
    du[5] =  k1*u4 - k2*u5

    nothing
end

u0s = [0.5,0.5,0.5,0.6,1.5]
params = [4,1,0.5,2,2,2,2,0.2,0.76,0.65,0.38,3.2,1.58,5,2.5,1.9,1.3,0.95]
prob = ODEProblem(Goldbeter!,u0s,(0.0,240),params)
sol = solve(prob)
plot(sol,tspan=(0,240))

###############################################################################
# Loading packages 
###############################################################################

function day(t)
    cos(2*3.14*(t/24)+5)+1
end 

function GoldbeterDay!(du,u,p,t)
    
    u1,u2,u3,u4,u5= u
    n,KI,Km,K1,K2,K3,K4,Kd,vs,vm,ks,V1,V2,V3,V4,k1,k2,vd = p
    
    du[1] = f0(vs,K1,KI,n,u5) - f(vm,Km,u1) + day(t)*exp(-u1)
    du[2] = ks*u1 - f(V1,K1,u2) + f(V2,K2,u3) - day(t)*u2
    du[3] =  f(V1,K1,u1) - f(V2,K2,u3) - f(V3,K3,u3) + f(V4,K4,u4)
    du[4] =  f(V3,K3,u3) - f(V4,K4,u4) - k1*u4 + k2*u5 - f(vd,Kd,u4)
    du[5] =  k1*u4 - k2*u5

    nothing
end

u0s = [0.5,0.5,0.5,0.6,1.5]
params = [4,1,0.5,2,2,2,2,0.2,0.76,0.65,0.38,3.2,1.58,5,2.5,1.9,1.3,0.95]
prob = ODEProblem(GoldbeterDay!,u0s,(0.0,240),params)
sol = solve(prob)
plot(sol,tspan=(0,240))

###############################################################################
# Loading packages 
###############################################################################

function day_shift(t)

    cos(2*3.14*(t/(24+0.02*t)))+1
end 

function GoldbeterDayShift!(du,u,p,t)
    
    u1,u2,u3,u4,u5= u
    n,KI,Km,K1,K2,K3,K4,Kd,vs,vm,ks,V1,V2,V3,V4,k1,k2,vd = p
    
    du[1] = f0(vs,K1,KI,n,u5) - f(vm,Km,u1) + day_shift(t)*exp(-u1)
    du[2] = ks*u1 - f(V1,K1,u2) + f(V2,K2,u3) - day_shift(t)*u2
    du[3] =  f(V1,K1,u1) - f(V2,K2,u3) - f(V3,K3,u3) + f(V4,K4,u4)
    du[4] =  f(V3,K3,u3) - f(V4,K4,u4) - k1*u4 + k2*u5 - f(vd,Kd,u4)
    du[5] =  k1*u4 - k2*u5

    nothing
end

u0s = [0.5,0.5,0.5,0.6,1.5]
params = [4,1,0.5,2,2,2,2,0.2,0.76,0.65,0.38,3.2,1.58,5,2.5,1.9,1.3,0.95]
prob = ODEProblem(GoldbeterDayShift!,u0s,(0.0,2400),params)
sol = solve(prob)
plot(sol,tspan=(0,2400))

###############################################################################
# Loading packages 
###############################################################################

function day_shift(t)

    (t*0.01)*cos(2*3.14*(t/24))+1
end 

function GoldbeterDayShift!(du,u,p,t)
    
    u1,u2,u3,u4,u5= u
    n,KI,Km,K1,K2,K3,K4,Kd,vs,vm,ks,V1,V2,V3,V4,k1,k2,vd = p
    
    du[1] = f0(vs,K1,KI,n,u5) - f(vm,Km,u1) + day_shift(t)*exp(-u1)
    du[2] = ks*u1 - f(V1,K1,u2) + f(V2,K2,u3) - day_shift(t)*u2
    du[3] =  f(V1,K1,u1) - f(V2,K2,u3) - f(V3,K3,u3) + f(V4,K4,u4)
    du[4] =  f(V3,K3,u3) - f(V4,K4,u4) - k1*u4 + k2*u5 - f(vd,Kd,u4)
    du[5] =  k1*u4 - k2*u5

    nothing
end

u0s = [0.5,0.5,0.5,0.6,1.5]
params = [4,1,0.5,2,2,2,2,0.2,0.76,0.65,0.38,3.2,1.58,5,2.5,1.9,1.3,0.95]
prob = ODEProblem(GoldbeterDayShift!,u0s,(0.0,2400),params)
sol = solve(prob)
plot(sol,tspan=(0,180))

###############################################################################
# Loading packages 
###############################################################################

function darklight(t)

    day = floor(t/24)
    hours = t-(day*24)
    if hours>6 && hours<18
        out = (1-cos(2*3.14*(t/24)))
    else
        out = 0.01
    end
end

function day_shift(t)
    dayoftheyearw = 1-sin(2*3.14*(t/(365*24))+(65*24))
    out = dayoftheyearw*darklight(t)
end 

function GoldbeterDayShift!(du,u,p,t)
    
    u1,u2,u3,u4,u5= u
    n,KI,Km,K1,K2,K3,K4,Kd,vs,vm,ks,V1,V2,V3,V4,k1,k2,vd = p
    
    du[1] = f0(vs,K1,KI,n,u5) - f(vm,Km,u1) + day_shift(t)*exp(-u1)
    du[2] = ks*u1 - f(V1,K1,u2) + f(V2,K2,u3) - day_shift(t)*u2
    du[3] =  f(V1,K1,u1) - f(V2,K2,u3) - f(V3,K3,u3) + f(V4,K4,u4)
    du[4] =  f(V3,K3,u3) - f(V4,K4,u4) - k1*u4 + k2*u5 - f(vd,Kd,u4)
    du[5] =  k1*u4 - k2*u5

    nothing
end

u0s = [0.5,0.5,0.5,0.6,1.5]
params = [4,1,0.5,2,2,2,2,0.2,0.76,0.65,0.38,3.2,1.58,5,2.5,1.9,1.3,0.95]
prob = ODEProblem(GoldbeterDayShift!,u0s,(0.0,8760),params)
sol = solve(prob)
plot(sol,tspan=(0,24*360))
