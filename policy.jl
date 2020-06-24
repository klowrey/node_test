
using DiffEqFlux, Flux, Optim, OrdinaryDiffEq


dstate = 4
dreward = 1

u0 = [0.0f0, 0.0f0, 1.001f0, 0.0f0, 0.0f0] # near bottom
#u0 = [0.0f0, 0.0f0, 0.05f0, 0.0f0, 0.0f0] # near TDC 
#u0 = [1.0f0, 0.0f0, 0.05f0, 0.0f0, 0.0f0] # near TDC 

T = 3.0f0
tspan = (0.0f0, T)
tsteps = 0.0f0:0.1f0:T

nh = 32
policy = FastChain(FastDense(4, nh, tanh),
                   #FastDense(nh, nh, tanh),
                   FastDense(nh, 1))
#policy = FastChain(FastDense(4, 1)) # linear policy

# The model weights are destructured into a vector of parameters
p_model = initial_params(policy)
p_model .*= 10

#θ = Float32[u0; p_model] # swingup
θ = p_model

function cartpole!(du, u, p, t)

    x, x_dot, theta, theta_dot = u

    #force = clamp(policy(u[1:dstate], p)[1], -10, 10) # TODO correct way to clamp?
    force = policy(u[1:dstate], p)[1]
    sintheta, costheta = sincos(theta)
    
    # self.polemass_length = 0.1 * 0.5 = 0.05 self.total_mass=1.0+0.1=1.1
    # self.gravity = 9.8
    temp     = (force + 0.05 * theta_dot^2 * sintheta) / 1.1
    thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (4.0 / 3.0 - 0.1 * costheta^2 / 1.1))
    xacc     = temp - 0.05 * thetaacc * costheta / 1.1

    # self.tau = 0.02   EULER default
    du[1] = x_dot                    # state
    du[2] = xacc
    du[3] = theta_dot
    du[4] = thetaacc
    du[5] = 0.01 * (x^2) - costheta # cost
end

prob_univ = ODEProblem(cartpole!, u0, tspan, p_model)

function predict_univ(θ)
    return Array(solve(prob_univ, Tsit5(), u0=u0, p=θ, saveat=tsteps))
end

#loss_univ(θ) = sum(abs2, predict_univ(θ)[5,:]) # or could do the reward function here?
loss_univ(θ) = predict_univ(θ)[5,end] # or could do the reward function here?
l = loss_univ(θ)
callback = function(θ, l)
  println(l)

  plt = lineplot(tsteps, -cos.(predict_univ(θ)[3,:]), name="theta",
                 xlim=tspan, #ylim = (-pi, pi),
                 width=60, height=6)
  #plt = lineplot(predict_univ(θ)[5,:], name="cost")
  display(plt)
  return false
end

#@time res1 = DiffEqFlux.sciml_train(loss_univ, θ, ADAM(0.03), cb = callback, maxiters=1000)
@time res1 = DiffEqFlux.sciml_train(loss_univ, θ,  LBFGS(), cb = callback)
