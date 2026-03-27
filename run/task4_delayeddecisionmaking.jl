using Random
using .Reservoirs_src

# Parameters
num_trials = 200         # Number of trials
stim_duration = 50       # Duration of stimulus presentation
resp_duration = 30       # Duration of response period
variance = 2.          # Variance of the sensory evidence

# Generate data
training_inputs, training_target_output = Reservoirs_src.generate_delay_decisionmaking_task_data(num_trials, stim_duration, resp_duration, variance)

plot(training_inputs[1,1:300])
# training_target_output = Float64.(training_target_output)

input_dim = 2
output_dim = 1
regularization_coefficient = 1e-6
input_scaling = 0.1
res_size = 300
sparsity = 0.1
spectral_radius = 0.99
leak_rate = 0.5

# Reservoir and input/output layers
reservoir = Reservoirs_src.create_reservoir(res_size, sparsity, spectral_radius)
res_size= size(reservoir)[1]
input_weights = Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
output_weights = Reservoirs_src.initialize_output_weights(res_size, output_dim)

reservoir_state_trains = zeros(res_size,size(training_inputs)[2])
current_state = zeros(res_size) 

for t in 1:size(training_inputs)[2]
    if t % (stim_duration + resp_duration) == 0
        reservoir_state_trains[:, t] = zeros(res_size)  # Reset state after each trial
    else
        current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, training_inputs[:,t], leak_rate)
        reservoir_state_trains[:, t] = current_state
    end
end


output_weights = Reservoirs_src.train_output_weights(reservoir_state_trains, training_target_output', regularization_coefficient)
training_output = output_weights' * reservoir_state_trains

plot(training_output[1:100])
plot!(training_target_output[1:100])

p1 = plot(training_target_output[302:400],lw=6, color=:blue, legend=false,xaxis=false,yaxis=false,grid=false)


# Generate test data
test_inputs, test_target_output = Reservoirs_src.generate_delay_decisionmaking_task_data(num_trials, stim_duration, resp_duration,variance)
#
test_inputs

last_state = reservoir_state_trains[:, end]
final_outputs = zeros(output_dim, size(test_inputs)[2])

reservoir_internal_state_ddm = zeros(res_size, size(test_inputs)[2])

for t in 1:size(test_inputs)[2]
    if t % (stim_duration + resp_duration) == 0
        last_state = zeros(res_size)  # Reset state after each trial
            reservoir_internal_state_ddm[:, t] = last_state
            predicted_output = output_weights' * last_state
            final_outputs[t] = predicted_output
    else
        next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_inputs[:,t], leak_rate)

        reservoir_internal_state_ddm[:, t] = next_state
        predicted_output = output_weights' * next_state

        # Storing the predicted output
        final_outputs[t] = predicted_output
        
        # Updating the last state for the next prediction
        last_state = next_state
    end
    
end

reservoir_internal_state_ddm


ddmplot = plot(test_target_output[100:550],lw=3,label="True", title="Output")
plot!(final_outputs[100:550], grid=false, c=:red,lw=3,label="Network output", yticks=[-1,0,1],ylim=(-2,2),xlabel="Time")


ddmp = plot(test_inputs[1,1:550],title="Input",legend=false,color=:violet,lw=2)

ddm_final = plot(ddmp,ddmplot,layout=(2,1),xlabelfontsize=14, legendfontsize=10, bottommargin=5mm, tick_direction=:out,grid=false,size=(600,300))


ddm_score = cor(test_target_output,final_outputs').^2
ddm_score = ddm_score[1,1]

function decision_accuracy_ignore_fixation1(y_pred, y_true, trial_len, fix_len; threshold=0.0)
    n_trials = length(y_true) ÷ trial_len
    correct = 0
    total = 0
    for i in 0:(n_trials-1)
        # Indices of decision period for trial i
        idx_start = i*trial_len + fix_len + 1
        idx_end = (i+1)*trial_len
        pred_segment = y_pred[idx_start:idx_end]
        true_segment = y_true[idx_start:idx_end]
        pred_labels = pred_segment .>= threshold
        true_labels = true_segment .>= threshold
        correct += sum(pred_labels .== true_labels)
        total += length(true_labels)
    end
    return correct / total
end

function decision_accuracy_ignore_fixation(y_pred, y_true, trial_len, fix_len)
    n_trials = length(y_true) ÷ trial_len
    correct = 0
    for i in 0:(n_trials - 1)
        idx_start = i * trial_len + fix_len + 1
        idx_end = (i + 1) * trial_len
        pred_mean = mean(y_pred[idx_start:idx_end])
        true_label = y_true[idx_end]  # assuming constant label in response window
        if sign(pred_mean) == sign(true_label)
            correct += 1
        end
    end
    return correct / n_trials
end

trial_len = stim_duration + resp_duration  # or whatever full trial length is
fix_len = stim_duration  # fixation/stimulus period to ignore
acc = decision_accuracy_ignore_fixation(final_outputs[:], test_target_output, trial_len, fix_len)
acc = decision_accuracy_ignore_fixation1(final_outputs[:], test_target_output, trial_len, fix_len)





function compute_error_rate_ignore_fixation(true_labels, predicted_outputs)
    # Convert outputs to ±1
    predicted_labels = sign.(predicted_outputs)

    # Find indices where true labels are NOT 0
    valid_indices = findall(x -> x != 0, true_labels)

    # Extract only the valid parts
    filtered_true = true_labels[valid_indices]
    filtered_predicted = predicted_labels[valid_indices]

    # Compute misclassification rate
    incorrect = sum(filtered_predicted .!= filtered_true)
    error_rate = incorrect / length(filtered_true)

    return error_rate
end

function ddm_nrmse_ignore_fixation(y_pred::AbstractVector, y_true::AbstractVector, trial_len::Int, fix_len::Int)
    n_trials = length(y_true) ÷ trial_len
    errors = Float64[]
    trues = Float64[]
    for i in 0:(n_trials-1)
        idx_start = i*trial_len + fix_len + 1
        idx_end = (i+1)*trial_len
        append!(errors, y_pred[idx_start:idx_end] .- y_true[idx_start:idx_end])
        append!(trues, y_true[idx_start:idx_end])
    end
    rmse = sqrt(mean(errors .^ 2))
    norm_factor = std(trues)
    return rmse / norm_factor
end


# Compute and print the modified error rate
error_rate_filtered = ddm_nrmse_ignore_fixation(vec(test_target_output), vec(final_outputs'), trial_len, fix_len)
