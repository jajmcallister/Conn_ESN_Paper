
using .Reservoirs_src
using LinearAlgebra, SparseArrays, OrdinaryDiffEq
using Statistics, Plots
using Arrow, DataFrames, CSV
using Dates
using FileIO


lorenz_savetimestep=0.02
prob = ODEProblem(Reservoirs_src.lorenz!, [1.0, 0.0, 0.0], (0.0, 200.0));
lorenz_data = solve(prob, ABM54(), dt=lorenz_savetimestep) 
lorenz_data = reduce(hcat, lorenz_data.u)
lorenz_data = lorenz_data[:,300:end]

train_period_length= 2000
test_period_length = 1000
lorenz_train_data, lorenz_test_data = Reservoirs_src.create_training_and_testing_periods(lorenz_data, train_period_length, test_period_length)

plot(lorenz_train_data[1,1:end])
real3d = plot3d(lorenz_train_data[1,:],lorenz_train_data[2,:],lorenz_train_data[3,:], c=:blue,label="Real Lorenz",legend=false,grid=false,title="Input Data",
xlabel="X", ylabel="Y", zlabel="Z", size=(800,500), titlefontsize=18, xlabelfontsize=16, ylabelfontsize=16, zlabelfontsize=16)

#parameters
input_size = 3
output_size = 3 
reservoir_size=300
spectral_radius = 0.99
sparsity= 0.1
leak_rate = 0.5
input_scaling = 0.05
regularization_coefficient = 1e-6

lorenztraindata = lorenz_train_data[:,1:end-1]
lorenztargetdata = lorenz_train_data[:,2:end]


##
reservoir = Reservoirs_src.create_reservoir(reservoir_size, sparsity, spectral_radius);

input_weights = Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_size);
reservoir_states = zeros(reservoir_size, size(lorenztraindata)[2])

# Initialising the reservoir
current_state = zeros(reservoir_size)   


# Training the output layer
for i in 1:size(lorenz_train_data, 2)-1
    current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, lorenztraindata[:, i], leak_rate)
    reservoir_states[:, i] = current_state
end

output_weights = Reservoirs_src.train_output_weights(reservoir_states, lorenztargetdata, regularization_coefficient)
trainingoutput = output_weights'*reservoir_states

plot(trainingoutput[1,800:end])
plot!(lorenz_train_data[1,800:end])

# Initialising predictions
predictions = zeros(output_size, test_period_length)

# Starting with the last state from training
last_state = reservoir_states[:, end]

lorenztestdata=lorenz_test_data[rand(2:6)]

reservoir_internal_states_lorenz = zeros(reservoir_size,test_period_length)

sum_network_activity = []

for i in 1:test_period_length
    # Generate the next state
    if i < 100
        input = lorenztestdata[:,i]
    else
        input = predictions[:, i-1]
    end
    next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
    push!(sum_network_activity, sum(abs.(next_state)))
    reservoir_internal_states_lorenz[:,i] = next_state
    predicted_output = output_weights' * next_state

    # Storing the predicted output
    predictions[:, i] = predicted_output
    
    # Updating the last state for the next prediction
    last_state = next_state
end

edit_test_data = lorenztestdata[:,99:999]
edit_predictions = predictions[:,100:end]

# measure of performance
threshhold = .5;
lorenztestdata[:,1:end-2]
predictions[:,2:end]
t1, l1 = Reservoirs_src.valid_time_lorenz(threshhold, predictions, lorenztestdata[:,1:end-1], lorenz_savetimestep)

#Plotting 
predictions
ts = 0.0:lorenz_savetimestep:100.0
lorenz_maxlyap = 0.9056
lyap_ts = ts*(1/lorenz_maxlyap)

p1 = plot(lyap_ts[1:800], edit_predictions[1,1:800], yticks=[-10,0,10], color="red", label="Prediction")
plot!(lyap_ts[1:800],edit_test_data[1,1:800], color="blue", label ="True", legend=:right)
vline!([l1], lw=3, label=false)
p2 = plot(lyap_ts[1:800], edit_predictions[2,1:800],color="red",  label=false)
plot!(lyap_ts[1:800],edit_test_data[2,1:800], color="blue", label =false)
vline!([l1],lw=3,label=false)
p3 = plot(lyap_ts[1:800],edit_predictions[3,1:800],color="red",  label=false)
plot!(lyap_ts[1:800],edit_test_data[3,1:800], color="blue",label =false)
vline!([l1],lw=3, label=false, xlabel="Time")
plot!([NaN],[NaN], lw=3, label="Valid Prediction Time", color=:green)
threeplots = plot(p1, p2, p3, layout=(3,1))

plot(vec(Reservoirs_src.calculate_nmse(predictions, lorenztestdata[:,2:end])))

real3d = plot3d(lorenz_train_data[1,:],lorenz_train_data[2,:],lorenz_train_data[3,:], c=:blue,label="Real Lorenz",legend=false)
plot3d!(edit_predictions[1,:],edit_predictions[2,:],edit_predictions[3,:], c=:red, label="Predicted Lorenz", legend=false)
lorenzplot = plot(real3d, threeplots, grid=false, layout=(1,2),
size=(800,500), plot_title="Lorenz Prediction",margin=1mm, tick_direction=:out,legendfontsize=12)







function compute_nrmse_lorenz(true_output, predicted_output)
    # Ensure the data is a matrix (reshape if needed)
    true_output = reshape(true_output, size(true_output, 1), :)
    predicted_output = reshape(predicted_output, size(predicted_output, 1), :)

    # Compute RMSE for each dimension separately
    error = true_output .- predicted_output
    rmse = sqrt.(mean(error .^ 2, dims=2))  
    
    # Normalize by the standard deviation of the true output per dimension
    norm_factor = std(true_output, dims=2) .+ 1e-8  # Small epsilon to avoid division by zero
    nrmse_per_dim = rmse ./ norm_factor

    return mean(nrmse_per_dim)  # Average across all dimensions
end

# Compute and print NRMSE for the Lorenz test data
nrmse_value = compute_nrmse_lorenz(edit_test_data, edit_predictions)


