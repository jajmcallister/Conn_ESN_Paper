using .Reservoirs_src
using .ReservoirTasks, .ConnectomeFunctions
using LinearAlgebra, SparseArrays, OrdinaryDiffEq
using Statistics, Plots, Plots.PlotMeasures
using DataFrames, CSV
using Dates
using FileIO


#parameters
reservoir_size = 300
sparsity = .1
spectral_radius = 0.999
input_scaling = 0.02
input_size = 1
output_size = 200  # N output neurons
regularization_coefficient = 1e-6

train_length = 4000
test_length = 1000
leak_rate = 1.0


# Create reservoir
reservoir = Reservoirs_src.create_reservoir(reservoir_size, sparsity, spectral_radius)
# Create input weights
input_weights = Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)

# Initialize output weights
output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

# Generate random input sequence
X_train = rand(train_length) .- 0.5
X_test = rand(test_length) .- 0.5

maximum(X_train)

# Generate target data for training the output layer
memory_target_data = Reservoirs_src.generate_memory_target_data(X_train, output_size)'


# Update reservoir state for training
reservoir_state_train = zeros(reservoir_size, train_length)
current_state = zeros(reservoir_size)   

for t in 1:train_length
    current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, X_train[t], leak_rate)
    reservoir_state_train[:, t] = current_state
end

# Train output weights
output_weights = Reservoirs_src.train_output_weights(reservoir_state_train, memory_target_data, regularization_coefficient)

s = vec(sum(abs.(output_weights),dims=2))

sortperm(s, rev=true)

# reservoir_state_test = zeros(reservoir_size, test_length)
final_outputs = zeros(output_size, test_length)


# Starting with the last state from training
last_state = reservoir_state_train[:, end]

reservoir_internal_states_memory = zeros(reservoir_size,test_length)

for t in 1:test_length
    next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, X_test[t], leak_rate)
    reservoir_internal_states_memory[:,t]=next_state
    predicted_output = output_weights' * next_state
    # Storing the predicted output
    final_outputs[:, t] = predicted_output
    
    # Updating the last state for the next prediction
    last_state = next_state
end


plot(reservoir_internal_states_memory[1:10,1])

X = vcat([X_train, X_test]...)

x_output = Reservoirs_src.generate_memory_target_data(X, output_size)'
test_output = x_output[:,train_length+1:end]


memory_capacity = 0
correlations = []
# Working out measure of memory capacity
for i in 1:output_size

    # Calculate squared Pearson correlation coefficient
    rho = cor(test_output[i,:], final_outputs[i,:])^2
    push!(correlations, rho)

    # Accumulate MC score
    memory_capacity += rho

end

memory_capacity

truedata = heatmap(test_output[1:50,1:100],title="True", ylabel="Output Neurons", xlabel="Time", size=(1000,1000),leftmargin=5mm, bottommargin=5mm)#, titlefont=titlefont)
predata = heatmap(final_outputs[1:50,1:100], title="Reservoir output", xlabel="Time",size=(1000,1000), bottommargin=5mm)#, titlefont=titlefont)

memoryplot = plot(truedata, predata, xtickfontsize=12, titlefontsize=18,clim=(-0.5,0.5), xlabelfontsize=16, ylabelfontsize=16, ytickfontsize=12,layout=(1,2), size=(1000,400)) #, leftmargin=5mm, bottommargin=5mm)



