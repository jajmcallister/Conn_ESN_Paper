using Random, Statistics, Plots
using .Reservoirs_src




# Example usage
num_samples_train = 2000   # Number of training samples
num_samples_test = 20000   # Number of testing samples
switch_interval = 100
input_dim = 2
output_dim = 1
regularization_coefficient = 1e-6
input_scaling = 0.1
res_size = 10
sparsity = 0.1
spectral_radius = 0.99
leak_rate = 0.01
bias = 0.5


# Generate training data
dm_train_data, dm_train_targets = Reservoirs_src.generate_decisionmaking_data(num_samples_train, switch_interval, bias)
dm_test_data, dm_test_targets = Reservoirs_src.generate_decisionmaking_data(num_samples_test, switch_interval, bias)

dm_test_data
plot(dm_train_targets[:])
plot!(dm_train_data[1,:] .- dm_train_data[2,:])

# Reservoir and input/output layers
reservoir = Reservoirs_src.create_reservoir(res_size, sparsity, spectral_radius)
reservoir = store_nets[20]
res_size= size(reservoir)[1]
input_weights = Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
output_weights = Reservoirs_src.initialize_output_weights(res_size, output_dim)


reservoir_state_trains = zeros(res_size,size(dm_train_data)[2])
current_state = zeros(res_size) 

for t in 1:size(dm_train_data)[2]
    if t % switch_interval == 0
        reservoir_state_trains[:, t]  = zeros(res_size)  # Reset state after each switch
    else
        current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, dm_train_data[:,t], leak_rate)
        reservoir_state_trains[:, t] = current_state
    end
end

reservoir_state_trains
count(!iszero, reservoir)

output_weights = Reservoirs_src.train_output_weights(reservoir_state_trains, dm_train_targets', regularization_coefficient)
training_output = output_weights' * reservoir_state_trains

plot(dm_train_targets[1:1000])
plot!(training_output[1:1000])


last_state = reservoir_state_trains[:, end]
final_outputs = zeros(output_dim, size(dm_test_data)[2])

reservoir_internal_states_dm = zeros(res_size, size(dm_test_data)[2])

for t in 1:size(dm_test_data)[2]
    if t % switch_interval == 0
        last_state = zeros(res_size)  # Reset state
    end
    next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, dm_test_data[:,t], leak_rate)

    predicted_output = output_weights' * next_state
    reservoir_internal_states_dm[:, t] = next_state

    # Storing the predicted output
    final_outputs[t] = predicted_output
    
    # Updating the last state for the next prediction
    last_state = next_state
end

using Plots.PlotMeasures    
pplot = plot(dm_test_targets[3:2000],lw=3,label="True", title="Two Input Decision Making Task",)
plot!(final_outputs[3:2000],lw=2,legend=:bottomright,tick_direction=:out,grid=false, margin=5mm, titlefontsize=16, xtickfontsize=14, ytickfontsize=14,  xlabelfontsize=14, legendfontsize=14, fillalpha=0.5, yticks=[-1,0,1], label="Reservoir output",xlabel="Time",size=(800,400))
# plot!(sign.(final_outputs[3:2000]),ls=:dash,lw=2,label="Signed output",c=:black)

dm_score = cor(dm_test_targets,final_outputs').^2
dm_score = dm_score[1,1]

function compute_accuracy(predicted_output, true_output; threshold=0.0)
    predicted_labels = sign.(predicted_output .- threshold)
    true_labels = sign.(true_output .- threshold)
    correct = sum(predicted_labels .== true_labels)
    return correct / length(true_output)
end

function compute_accuracy(predicted_output, true_output; threshold=0.0)
    predicted_labels = ifelse.(predicted_output .>= threshold, 1, -1)
    true_labels      = ifelse.(true_output .>= threshold, 1, -1)
    correct = sum(predicted_labels .== true_labels)
    return correct / length(true_output)
end


dm_score = compute_accuracy(final_outputs, dm_test_targets', threshold=0.0)


function compute_dm_error_rate(true_labels, predicted_outputs)
    predicted_labels = sign.(predicted_outputs)  # Convert outputs to ±1
    incorrect = sum(predicted_labels .!= true_labels)  # Count misclassifications
    error_rate = incorrect / length(true_labels)  # Compute error rate
    return error_rate
end

function compute_dm_nrmse(y_true, y_pred)
    error = y_true .- y_pred
    rmse = sqrt(mean(error .^ 2))
    norm_factor = std(y_true) + 1e-8  # avoid division by zero
    return rmse / norm_factor
end


# Compute and print the error rate
error_rate = compute_dm_error_rate(dm_test_targets, final_outputs')
error_dm = compute_dm_nrmse(dm_test_targets, final_outputs')




function interval_accuracy1(predicted_output,
                           true_output,
                           interval_length::Int;
                           threshold::Float64=0.0)

    n_intervals = length(predicted_output) ÷ interval_length
    correct_intervals = 0

    for i in 0:(n_intervals-1)
        start_idx = i*interval_length + 1
        end_idx = start_idx + interval_length - 1

        pred_interval = predicted_output[start_idx:end_idx]
        true_interval = true_output[start_idx:end_idx]

        # majority vote in the interval
        s = sum(sign.(pred_interval .- threshold))
        if s == 0
            # if all zeros, randomly pick +1 or -1 (or consider it 50% correct)
            pred_label = rand(Bool) ? 1 : -1
        else
            pred_label = sign(s)
        end

        # majority of true interval
        ts = sum(sign.(true_interval .- threshold))
        true_label = ts == 0 ? 1 : sign(ts)  # should not be 0, just in case

        if pred_label == true_label
            correct_intervals += 1
        end
    end

    return correct_intervals / n_intervals
end


interval_accuracy1(final_outputs', dm_test_targets', 100, threshold=0.0)


