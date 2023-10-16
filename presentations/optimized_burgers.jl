# Slow version
using OrdinaryDiffEq, BenchmarkTools

const Nx = 1024
function wrap(i)
    if i < 1
        return i + Nx
    elseif i > Nx
        return i - Nx
    else
        return i
    end
end

function burgers(du, u, p, t)
    dx, Nx, v = p
    
    for i in 1:Nx
        if u[i] < 0 
            du[i] = -u[i]*(u[wrap(i+1)] - u[i])/dx + v*(u[wrap(i-1)] - 2*u[i] + u[wrap(i+1)])/dx^2
        elseif u[i] > 0
            du[i] = -u[i]*(u[i] - u[wrap(i-1)])/dx + v*(u[wrap(i-1)] - 2*u[i] + u[wrap(i+1)])/dx^2
        else
            du[i] = v*(u[wrap(i-1)] - 2*u[i] + u[wrap(i+1)])/dx^2
        end
    end
    nothing
end
dx = 2/(Nx -1)
x =-1.0:dx:1.0
v = 0.01
p = (dx, Nx, v)
sinsin(x) = 0.11*cospi(4x)+0.75
u0 = sinsin.(x)
prob = ODEProblem{true, SciMLBase.FullSpecialize}(burgers, u0, (0.0, 1.0), p)

sol = solve(prob, Euler(), reltol=1e-4, saveat=0.01, dt=1e-4);
@btime solve(prob, Euler(), reltol=1e-4, saveat=0.01, dt=1e-4);

using Plots
anim = @animate for i in eachindex(sol.t)
    plot(x, sol.u[i], title = "t = $(sol.t[i])")
end
gif(anim, "pdebenchburgers.gif", fps = 20)

# Optimized version

using OrdinaryDiffEq, BenchmarkTools, LoopVectorization, LinearSolve

const Nx = 1024
const dx = 2/(Nx -1)
function burgers(du, u, p, t)
    v = p
    
    @simd ivdep for i in 2:Nx-1
      du[i] = -u[i]*(u[i] - u[i-1])/dx + v*(u[i-1] - 2*u[i] + u[i+1])/dx^2
    end
    error()

    @inbounds du[1] = -u[1]*(u[1] - u[end])/dx + v*(u[end] - 2*u[1] + u[2])/dx^2
    @inbounds du[end] = -u[end]*(u[end] - u[end-1])/dx + v*(u[end-1] - 2*u[end] + u[1])/dx^2
    nothing
end

x =-1.0:dx:1.0
v = 0.01
p = v
sinsin(x) = 0.11*cospi(4x)+0.75
u0 = sinsin.(x)
__f = ODEFunction(burgers, jac_prototype = Tridiagonal(zeros(Nx-1), zeros(Nx), zeros(Nx-1)))
prob = ODEProblem{true, SciMLBase.FullSpecialize}(__f, u0, (0.0, 2.0), p)

sol = solve(prob, ROCK2(eigen_est = (integ)-> integ.eigen_est = 12500), reltol=1e-4, saveat=0.01, unstable_check = (dt,u,p,t)-> false);

using Plots
anim = @animate for i in eachindex(sol.t)
    plot(x, sol.u[i], title = "t = $(sol.t[i])")
end
gif(anim, "pdebenchburgers.gif", fps = 20)

bprob = ODEProblem{true, SciMLBase.FullSpecialize}(__f, sol[end], (2.0, 0.0), p)
sol = solve(bprob, ROCK2(eigen_est = (integ)-> integ.eigen_est = 12500), reltol=1e-4, saveat=0.01, unstable_check = (dt,u,p,t)-> false);

anim = @animate for i in eachindex(sol.t)
  plot(x, sol.u[i], title = "t = $(sol.t[i])")
end
gif(anim, "pdebenchburgers.gif", fps = 20)


@btime sol = solve(prob, Euler(), reltol=1e-4, saveat=0.01, unstable_check = (dt,u,p,t)-> false, dt = 1e-4);

@btime sol = solve(prob, ROCK2(eigen_est = (integ)-> integ.eigen_est = 12500), reltol=1e-4, saveat=0.01, unstable_check = (dt,u,p,t)-> false);

integ = init(prob, ROCK2(eigen_est = (integ)-> integ.eigen_est = 12500), reltol=1e-4, saveat=0.01, unstable_check = (dt,u,p,t)-> false);

function resolve!(integ)
  reinit!(integ; erase_sol=false)
  solve!(integ)
end

@btime resolve!(integ);



@profview for i in 1:100 resolve!(integ); end



data = Array(solve(prob, ROCK2(), reltol=1e-4, saveat = 0.1))

function _loss(p, data)
  _prob = remake(prob, p=p)
  sol = solve(_prob, ROCK2(eigen_est = (integ)-> integ.eigen_est = 12500), reltol=1e-6, 
                          saveat=0.1, unstable_check = (dt,u,p,t)-> false,
                          sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)));
  
  size(Array(sol)) != size(data) && return Inf
  diff = sol.-data
  sum(abs2, diff)
end

loss(p, _) = _loss(p, data)

using ForwardDiff, FiniteDiff, Zygote, Optimization, OptimizationOptimJL, SciMLSensitivity
function burgers(du, u, p, t)
  v = p[1]
  
  @simd ivdep for i in 2:Nx-1
    du[i] = -u[i]*(u[i] - u[i-1])/dx + v*(u[i-1] - 2*u[i] + u[i+1])/dx^2
  end

  @inbounds du[1] = -u[1]*(u[1] - u[end])/dx + v*(u[end] - 2*u[1] + u[2])/dx^2
  @inbounds du[end] = -u[end]*(u[end] - u[end-1])/dx + v*(u[end-1] - 2*u[end] + u[1])/dx^2
  nothing
end

@time ForwardDiff.gradient(x->loss(x,nothing), [0.01])
@time Zygote.gradient(x->loss(x,nothing), [0.01])
@time FiniteDiff.finite_difference_gradient(x->loss(x,nothing), [0.01])


f = OptimizationFunction(loss, AutoForwardDiff())
optprob = OptimizationProblem(f, [0.5])
@time sol = solve(optprob, BFGS())


f = OptimizationFunction(loss, AutoZygote())
optprob = OptimizationProblem(f, [0.5])
@time sol = solve(optprob, BFGS())

sol = solve(prob, ROCK2(eigen_est = (integ)-> integ.eigen_est = 12500), reltol=1e-4, saveat=0.1, unstable_check = (dt,u,p,t)-> false);

#0.021883010864257812 sec / 130.000 μs = 168.330853
