using .Reservoirs_src
using LinearAlgebra, SparseArrays, Plots, Statistics, OrdinaryDiffEq

# Define Lotka-Volterra system
function lotka_volterra!(du, u, p, t)
    α, β, δ, γ = p
    du[1] = α*u[1] - β*u[1]*u[2]
    du[2] = δ*u[1]*u[2] - γ*u[2]
end

# Parameters and initial conditions
u0 = [1.0, 1.0]
p = (1.5, 1.0, 1.0, 3.0)
tspan = (0.0, 1000.0)
dt = 0.1

prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=dt)
fulldata = reduce(hcat, sol.u)'  # shape: timepoints × 2

lotka_data = fulldata

# Split into train and test sets
train_period_length = 2000
test_period_length = 1000

train_data = fulldata[1:train_period_length, :]'
test_data = fulldata[train_period_length+1 : train_period_length+test_period_length, :]'

p1 = plot(train_data[1,2:301], color=:blue, title="Prey")
p2 = plot(train_data[2,2:301], label="True Predator", color=:green, title="Predator")
p = plot(p1, p2, layout=(2, 1), size=(800, 600), lw=3, grid=false,  legend=false, xlabel="Time", ylabel="Population",dpi=500)

# Parameters
input_size = 2
output_size = 2
reservoir_size = 300
spectral_radius = 0.99
sparsity = 0.01
leak_rate = 1.
input_scaling = 0.1
regularization_coefficient = 1e-6

# Create the reservoir
reservoir = Reservoirs_src.create_reservoir(reservoir_size, sparsity, spectral_radius)
input_weights = Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

# Initialize reservoir states
reservoir_states = zeros(reservoir_size, size(train_data, 2)-1)
current_state = zeros(reservoir_size)

# Training
for i in 1:size(train_data, 2)-1
    input = train_data[:, i]
    current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
    reservoir_states[:, i] = current_state
end

# Train output weights
output_targets = train_data[:, 2:end]
output_weights = Reservoirs_src.train_output_weights(reservoir_states, output_targets, regularization_coefficient)
training_output = output_weights' * reservoir_states

# Plot training prediction (first 1000)
p1 = plot(training_output[1,1:1000], label="Pred Prey", color=:red)
plot!(train_data[1,2:1001], label="True Prey", color=:blue, title="Prey")
p2 = plot(training_output[2,1:1000], label="Pred Predator", color=:orange)
plot!(train_data[2,2:1001], label="True Predator", color=:green, title="Predator")

plot(p1, p2, layout=(2, 1), size=(800, 600), plot_title="Training", xlabel="Time", ylabel="Population")

# Predict
predictions = zeros(output_size, test_period_length)
last_state = reservoir_states[:, end]

reservoir_internal_states_lot = zeros(reservoir_size, test_period_length)
for i in 1:test_period_length
    input = i < 100 ? test_data[:, i] : predictions[:, i-1]
    next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
    xx = output_weights' * next_state
    reservoir_internal_states_lot[:, i] = next_state
    predictions[:, i] = xx
    last_state = next_state
end

# Evaluate
pp = predictions[:, 1:end-1]
ppp = test_data[:, 2:end]


# NRMSE function
function compute_nrmse_lotka(true_output, predicted_output)
    error = true_output .- predicted_output
    rmse = sqrt.(mean(error .^ 2, dims=2))
    norm_factor = std(true_output, dims=2) .+ 1e-8
    nrmse_per_dim = rmse ./ norm_factor
    return mean(nrmse_per_dim)
end


# Compute NRMSE
nrmse_value = compute_nrmse_lotka(ppp, pp)

# Valid time
t1, l1 = Reservoirs_src.valid_time_lotka(0.5, pp, ppp, dt)

# Plot results
p1 = plot(predictions[1,100:end], label="Predicted Prey", color=:red)
plot!(test_data[1,101:end], label="True Prey", color=:blue, title="Prey")
vline!([t1], label="Valid time prediction", color=:green, lw=3)
p2 = plot(predictions[2,100:500], label="Predicted Predator", color=:orange)
plot!(test_data[2,101:501], label="True Predator", color=:purple, title="Predator")
vline!([t1], label=false, color=:green, lw=3,xlabel="Time", )

p3 = plot(p1, p2, layout=(2, 1), tick_direction=:out, grid=false, xlabelfontsize=14,ylabelfontsize=14,legendfontsize=14, xtickfontsize=12, ytickfontsize=12, size=(800, 600), plot_title="Lotka-Volterra Prediction", ylabel="Population")
