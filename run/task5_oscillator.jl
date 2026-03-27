using .Reservoirs_src
using LinearAlgebra, SparseArrays, Plots
using Statistics

# Define the function
function periodic_function(x)
    return 3*sin(x) + 2*cos(2x) + sin(2.3*x)
end

# Generate time series data
savetimestep = 0.1
x_values = 0:savetimestep:100000  # Time steps
fulldata = periodic_function.(x_values)

# Split into train and test sets
train_period_length = 2000
test_period_length = 5000

ii = rand(train_period_length:length(fulldata)-1)
test_data = fulldata[train_period_length+1+ii:train_period_length+ii+test_period_length]

osc_data = deepcopy(fulldata)

train_data = fulldata[1:train_period_length]


p = plot(train_data[1:500],legend=false,grid=false,lw=3,title="Input Data",c=:blue,size=(600,400),xlabel="Time")

# Parameters
input_size = 1
output_size = 1
reservoir_size = 300
spectral_radius = 0.99
sparsity = 0.05
leak_rate = .2
input_scaling = 0.01
regularization_coefficient = 1e-6

# Create the reservoir
reservoir = Reservoirs_src.create_reservoir(reservoir_size, sparsity, spectral_radius)
input_weights = Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

# Initialise reservoir states
reservoir_states = zeros(reservoir_size, length(train_data)-1)
current_state = zeros(reservoir_size)

# Training
for i in 1:length(train_data)-1
    input = train_data[i]  # Input must be a vector
    current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
    reservoir_states[:, i] = current_state
end

# Train output weights
output_weights = Reservoirs_src.train_output_weights(reservoir_states, train_data[2:end]', regularization_coefficient)
training_output = output_weights' * reservoir_states

plot(vcat(training_output...)[1:1000], label="Predicted", color=:red)
plot!(train_data[2:1002], label="True", color=:blue)

# Initialize predictions
predictions = zeros(output_size, test_period_length)
last_state = reservoir_states[:, end]

rr = rand(train_period_length:length(fulldata)-1)
test_data = fulldata[train_period_length+rr:train_period_length+rr+test_period_length-1]

reservoir_internal_states_osc = zeros(reservoir_size, test_period_length)
# Predict using only previous predictions after 100 steps
for i in 1:test_period_length
    if i < 100
        input = test_data[i]
    else
        input = predictions[i-1]
    end
    next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
    xx = output_weights' * next_state
    reservoir_internal_states_osc[:, i] = next_state
    predictions[i] = xx[1]
    last_state = next_state
end


pp = predictions[1:end-1]
ppp = test_data[2:end]

t1, l1 = Reservoirs_src.valid_time_oscillator(0.5, reshape(pp, 1, length(pp)), reshape(ppp, 1, length(ppp)), savetimestep)

default(fontfamily="JuliaMono")
# Plot Results
predictions1 = predictions[1,100:end]
plottest_data = test_data[100:end]
p = plot(predictions1[1:1000], label="Predicted", lw=2, color=:red)
plot!(plottest_data[2:1000], label="True",lw=2, color=:blue)
vline!([t1], label="Valid time prediction",tick_direction=:out, legend=:bottomleft, legendfontsize=12, titlefontsize=18, margin=7mm, xlabel="Time", xlabelfontsize=17, ytickfontsize=14, xtickfontsize=14, grid=false, color=:green, lw=3, title="Trigonometric Oscillator Prediction", size=(1000,400), dpi=500)


function compute_nrmse_oscillator(true_output, predicted_output)
    # Ensure inputs are matrices
    true_output = reshape(true_output, size(true_output, 1), :)
    predicted_output = reshape(predicted_output, size(predicted_output, 1), :)

    # Compute RMSE per dimension
    error = true_output .- predicted_output
    rmse = sqrt.(mean(error .^ 2, dims=2))

    # Normalize by std of true output
    norm_factor = std(true_output, dims=2) .+ 1e-8
    nrmse_per_dim = rmse ./ norm_factor

    return mean(nrmse_per_dim)
end

nrmse_value = compute_nrmse_oscillator(reshape(ppp,1,length(ppp)), reshape(pp,1,length(pp)))

