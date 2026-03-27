using .Reservoirs_src
using LinearAlgebra, SparseArrays, OrdinaryDiffEq
using Statistics, Plots
using Arrow, DataFrames, CSV
using Dates
using FileIO


# Define the Rössler system equations
function roessler!(du, u, p, t)
    a = 0.15
    b = 0.2
    c = 10.0
    du[1] = -u[2] - u[3]
    du[2] = u[1] + a * u[2]
    du[3] = b + u[3] * (u[1] - c)
end

# Parameters for the Rössler system
a = 0.15
b = 0.2
c = 10.0
p = (a, b, c)


u0 = [-1.0, 1.0, 1.0]
tspan = (0.0, 100000.0)

# Solve the system
rossler_savetimestep = 0.04
prob = ODEProblem(roessler!, u0, tspan)
rossler_sol = solve(prob, Tsit5(), saveat=rossler_savetimestep)
rossler_data = reduce(hcat, rossler_sol.u)
rossler_data = rossler_data[:,500:end]



train_period_length= 10000
test_period_length = 5000
rossler_train_data, rossler_test_data = Reservoirs_src.create_training_and_testing_periods(rossler_data, train_period_length, test_period_length)


rossler_train_data[:,1:end-1]
ross3d = plot3d(rossler_train_data[1,:],rossler_train_data[2,:],rossler_train_data[3,:],color=:blue,legend=false)

#parameters
input_size = 3
output_size = 3 
reservoir_size= 300
spectral_radius = 0.99
sparsity= 0.1
leak_rate = 1.0
input_scaling = 0.04
regularization_coefficient = 1e-6

rosslertraindata = rossler_train_data[:,1:end-1]
rosslertargetdata = rossler_train_data[:,2:end]


##
reservoir = Reservoirs_src.create_reservoir(reservoir_size, sparsity, spectral_radius);
input_weights = Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_size);
reservoir_states = zeros(reservoir_size, size(rosslertraindata)[2])

# Initialising the reservoir
current_state = zeros(reservoir_size)   


# Training the output layer
for i in 1:size(rossler_train_data, 2)-1
    current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, rosslertraindata[:, i], leak_rate)
    reservoir_states[:, i] = current_state
end

output_weights = Reservoirs_src.train_output_weights(reservoir_states, rosslertargetdata, regularization_coefficient)
trainingoutput = output_weights'*reservoir_states

plot3d(trainingoutput[1,:],trainingoutput[2,:],trainingoutput[3,:], color=:red, legend=false)
plot3d!(rossler_train_data[1,:],rossler_train_data[2,:],rossler_train_data[3,:], color=:blue, legend=false)

plot(trainingoutput[1,:])
plot!(rossler_train_data[1,:])

# Initialising predictions
predictions = zeros(output_size, test_period_length)

# Starting with the last state from training
last_state = reservoir_states[:, end]

rosslertestdata=rossler_test_data[1][:,1:end-1]

reservoir_internal_states_rossler = zeros(reservoir_size, test_period_length)
for i in 1:test_period_length
    # Generate the next state
    if i < 100
        input = rosslertestdata[:,i]
    else
        input = predictions[:, i-1]
    end
    next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
    predicted_output = output_weights' * next_state
    reservoir_internal_states_rossler[:,i] = next_state

    # Storing the predicted output
    predictions[:, i] = predicted_output
    
    # Updating the last state for the next prediction
    last_state = next_state
end

rosslertestdata
predictions
rosslertestdata
edit_test_data = rosslertestdata[:,99:end-1]
edit_predictions = predictions[:,100:end]

plot(edit_predictions[1,:])
plot!(edit_test_data[1,:],xlim=(0,100))

# measure of performance
threshhold = .5;


t1, l1 = Reservoirs_src.valid_time_rossler(threshhold, predictions, rosslertestdata[:,1:end], rossler_savetimestep)



#Plotting 

predictions
ts = 0.0:rossler_savetimestep:100.0
rossler_maxlyap = .09
ross_ts = ts*(1/rossler_maxlyap)

p1 = plot(ross_ts[1:700], edit_predictions[1,1:700], color="red", label="Prediction")
plot!(ross_ts[1:700],edit_test_data[1,1:700], color="blue", label ="True")
vline!([l1], label=false,lw=3)
p2 = plot(ross_ts[1:700], edit_predictions[2,1:700],color="red",  label=false)
plot!(ross_ts[1:700],edit_test_data[2,1:700], color="blue", label =false)
vline!([l1],label=false,lw=3)
p3 = plot(ross_ts[1:700],edit_predictions[3,1:700],color="red",  label=false)
plot!(ross_ts[1:700],edit_test_data[3,1:700], color="blue",label =false)
vline!([l1],label=false,lw=3)
plot!([NaN],[NaN],color=:green,lw=3,  label="Valid Prediction Time")

threeplots  = plot(p1, p2, p3, layout=(3,1))


real3d = plot3d(rossler_train_data[1,:],rossler_train_data[2,:],rossler_train_data[3,:], c=:blue,label="Real Rossler",legend=false)
plot3d!(edit_predictions[1,1:600],edit_predictions[2,1:600],edit_predictions[3,1:600], c=:red, label="Predicted Rossler", legend=false)
rosslerplot = plot(real3d, threeplots, grid=false, layout=(1,2),
size=(800,500), plot_title="Rossler Prediction",margin=1mm, tick_direction=:out,legendfontsize=12)


pp = plot(p1, p2, p3, layout=(3,1))
r = plot(ross3d,pp,layout=(1,2),size=(1000,300),grid=false)


function compute_nrmse_rossler(true_output, predicted_output)
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


compute_nrmse_rossler(edit_test_data,edit_predictions)