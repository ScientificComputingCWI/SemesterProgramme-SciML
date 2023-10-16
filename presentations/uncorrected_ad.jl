using SciMLSensitivity, OrdinaryDiffEq, Zygote

function get_param(breakpoints, values, t)
    for (i, tᵢ) in enumerate(breakpoints)
        if t <= tᵢ
            return values[i]
        end
    end

    return values[end]
end

function fiip(du, u, p, t)
    a = get_param([1., 2., 3.], p[1:4],  t)

    du[1] = dx =  a * u[1] - u[1] * u[2]
    du[2] = dy = -a * u[2] + u[1] * u[2]
end

p = [1., 1., 1., 1.]; u0 = [1.0;1.0]
prob = ODEProblem(fiip, u0, (0.0, 4.0), p);

solve(prob, Tsit5(), reltol=1e-6)
solve(prob, Tsit5(), reltol=1f-6)

# Original AD
Zygote.gradient(p->sum(solve(prob, Tsit5(), u0=u0, p=p, sensealg = ForwardDiffSensitivity(), saveat = 0.1, internalnorm = (u,t) -> sum(abs2,u/length(u)), abstol=1e-9, reltol=1e-9)), p
)

# Forward Sensitivity
Zygote.gradient(p->sum(solve(prob, Tsit5(), u0=u0, p=p, sensealg = ForwardSensitivity(), saveat = 0.1, abstol=1e-12, reltol=1e-12)), p
)

# Corrected AD
Zygote.gradient(p->sum(solve(prob, Tsit5(), u0=u0, p=p, sensealg = ForwardDiffSensitivity(), saveat = 0.1, abstol=1e-12, reltol=1e-12)), p
)
