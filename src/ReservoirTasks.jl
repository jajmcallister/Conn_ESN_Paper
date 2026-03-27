using .Reservoirs_src

module ReservoirTasks

using Random, LinearAlgebra, Statistics, StatsBase, OrdinaryDiffEq, ProgressLogging

export res_performance_delay_decisionmaking_L, res_performance_decisionmaking_L, res_performance_recall_L, res_performance_lorenz, res_performance_rossler, res_performance_memory, res_performance_recall, res_performance_dotmotion, res_performance_decisionmaking, res_performance_delay_decisionmaking, lorenz_pruning, rossler_pruning, memory_pruning, recall_pruning, decisionmaking_pruning, delay_decisionmaking_pruning

function create_data(L)
    x1 = vcat([rand() for i in 1:L],[0 for i in 1:L+1])
    x2 = vcat([0 for i in 1:L],[1],[0 for i in 1:L])
    x3 = vcat([0 for i in 1:L+1],x1[1:L])
    return x1,x2,x3
end

function compute_mem_nrmse(test_output, final_outputs)
    error = test_output .- final_outputs
    rmse = sqrt(mean(error .^ 2))
    norm_factor = std(test_output)
    return rmse / norm_factor
end

function compute_rec_nrmse(test_output, final_outputs, L) #ignoring fixation periods
    # Copy inputs to avoid modifying originals
    test_out = copy(test_output)
    final_out = copy(final_outputs)
    
    # Zero out fixation periods: 1st, 3rd, 5th, ... blocks of length L
    total_length = length(test_out)
    n_blocks = div(total_length, L)
    for block in 1:n_blocks
        if isodd(block)
            idx_start = (block-1)*L + 1
            idx_end = min(block*L, total_length)
            test_out[idx_start:idx_end] .= 0
            final_out[idx_start:idx_end] .= 0
        end
    end

    error = test_out .- final_out
    rmse = sqrt(mean(error .^ 2))
    norm_factor = std(test_out)
    return rmse / norm_factor
end

function compute_dm_error_rate(true_labels, predicted_outputs)
    predicted_labels = sign.(predicted_outputs)  # Convert outputs to ±1
    incorrect = sum(predicted_labels .!= true_labels)  # Count misclassifications
    error_rate = incorrect / length(true_labels)  # Compute error rate
    return error_rate
end


function compute_dm_accuracy(predicted_output, true_output; threshold=0.0)
    predicted_labels = ifelse.(predicted_output .>= threshold, 1, -1)
    true_labels      = ifelse.(true_output .>= threshold, 1, -1)
    correct = sum(predicted_labels .== true_labels)
    return correct / length(true_output)
end

function compute_dm_accuracy_window(predicted_output, true_output;
                                    decision_window=50, threshold=0.0)

    T = length(predicted_output)
    @assert T ≥ decision_window "Decision window longer than signal."

    # Average over decision window
    ŷ = mean(predicted_output[end-decision_window+1:end])
    y  = mean(true_output[end-decision_window+1:end])

    pred_label = ŷ ≥ threshold ? 1 : -1
    true_label = y ≥ threshold ? 1 : -1

    return pred_label == true_label ? 1.0 : 0.0
end


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


function compute_dm_nrmse(y_true, y_pred)
    error = y_true .- y_pred
    rmse = sqrt(mean(error .^ 2))
    norm_factor = std(y_true) + 1e-8  # avoid division by zero
    return rmse / norm_factor
end

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


function delay_decision_accuracy_ignore_fixation(y_pred, y_true, trial_len, fix_len; threshold=0.0)
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

function compute_nrmse_oscillator(true_output, predicted_output)
    # Ensure column vectors
    true_output = reshape(true_output, 1, :)
    predicted_output = reshape(predicted_output, 1, :)

    error = true_output .- predicted_output
    rmse = sqrt.(mean(error .^ 2, dims=2))

    norm_factor = std(true_output, dims=2) .+ 1e-8
    nrmse = rmse ./ norm_factor

    return nrmse[1]  # Since it's 1D, return scalar
end

function compute_nrmse_lotka2(true_output, predicted_output)
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

function compute_nrmse_lotka(true_output, predicted_output)
    error = true_output .- predicted_output
    rmse = sqrt.(mean(error .^ 2, dims=2))
    norm_factor = std(true_output, dims=2) .+ 1e-8
    nrmse_per_dim = rmse ./ norm_factor
    return mean(nrmse_per_dim)
end


function recall_accuracy_ignore_fixation(ŷ, y, L, ε=0.1)
    ŷ_mod = copy(ŷ)
    n_blocks = length(ŷ) ÷ L
    for i in 1:2:n_blocks  # 1st, 3rd, 5th, ... blocks (fixation periods)
        start_idx = (i - 1)*L + 1
        end_idx = i*L
        ŷ_mod[start_idx:end_idx] .= 0
    end
    
    # now compute accuracy ignoring the zeroed periods:
    mask = ŷ_mod .!= 0
    correct = abs.(ŷ_mod[mask] .- y[mask]) .< ε
    return sum(correct) / length(correct)
end

####

function res_performance_memory(reservoirs, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate)


    #parameters
    reservoir_size = size(reservoirs[1])[1]
    input_size = 1
    output_size = 100  # N output neurons
    train_length = 4000
    test_length = 1000

    memories = []
    total_performances = []
    total_errors = []

    @progress for reservoir in reservoirs
        mems = []

        for i in 1:num_simulation_steps
            mem = []

            # Create input and output weights
            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            # Generate random input sequence
            X_train = rand(train_length) .- 0.5
            X_test = rand(test_length) .- 0.5

            # Generate target data for training the output layer
            target_data = Main.Reservoirs_src.generate_memory_target_data(X_train, output_size)'


            # Update reservoir state for training
            reservoir_state_train = zeros(reservoir_size, train_length)

            current_state = zeros(reservoir_size)   

            for t in 1:train_length
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, X_train[t], leak_rate)
                reservoir_state_train[:, t] = current_state
            end

            # Train output weights
            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_train, target_data, regularization_coefficient)

            reservoir_state_test = zeros(reservoir_size, test_length)
            final_outputs = zeros(output_size, test_length)


            # Starting with the last state from training
            last_state = reservoir_state_train[:, end]

            for t in 1:test_length
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, X_test[t], leak_rate)

                predicted_output = output_weights' * next_state


                # Storing the predicted output
                final_outputs[:, t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            X = vcat([X_train, X_test]...)
            x_output = Main.Reservoirs_src.generate_memory_target_data(X, output_size)'
            test_output = x_output[:,4001:end]


            memory_capacity = 0
            # Working out measure of memory capacity
            for i in 1:output_size
                # Calculate squared Pearson correlation coefficient
                rho = cor(test_output[i,:], final_outputs[i,:])^2
                # Accumulate MC score
                memory_capacity += rho
            end

            push!(mem, memory_capacity)
            push!(total_performances, memory_capacity)

            # Compute and print NRMSE
            nrmse_value = compute_mem_nrmse(test_output, final_outputs)
            push!(total_errors, nrmse_value)


            push!(mems, mean(mem))
        end
        push!(memories, mean(mems))
    end

    return total_performances, total_errors

                    
end

function res_performance_recall_old(reservoirs, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate)


    #parameters
    res_size = size(reservoirs[1])[1]

    input_dim = 2
    output_dim = 1
    n_trials_train = 200  # Number of training trials
    n_trials_test = 50  # Number of testing trials
    
    recall_scores = []
    total_performances = []
    total_errors = []


    @progress for reservoir in reservoirs
        recalls1 = []

        for i in 1:num_simulation_steps

            L = 30
            # Create weights
            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)

            

            x1_train_data = []
            x2_train_data = []
            x3_train_targets = []

            for x in 1:n_trials_train
                x1,x2,x3 = create_data(L)
                push!(x1_train_data, x1)
                push!(x2_train_data, x2)
                push!(x3_train_targets,x3)
            end

            x1_train_data = vcat(x1_train_data...)
            x2_train_data = vcat(x2_train_data...)
            x3_train_targets = vcat(x3_train_targets...)


            input_data = hcat(x1_train_data,x2_train_data)
            input_data = input_data'

            input_weights*input_data[:,100]

            reservoir_state_trains = zeros(res_size,length(x1_train_data))
            current_state = zeros(res_size) 

            for t in 1:length(x1_train_data)
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input_data[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, x3_train_targets', regularization_coefficient)
            training_output = output_weights' * reservoir_state_trains

            x1_test_data = []
            x2_test_data = []
            x3_test_targets = []

            for x in 1:n_trials_test
                x1,x2,x3 = create_data(L)
                push!(x1_test_data, x1)
                push!(x2_test_data, x2)
                push!(x3_test_targets,x3)
            end

            x1_test_data = vcat(x1_test_data...)
            x2_test_data = vcat(x2_test_data...)
            x3_test_targets = vcat(x3_test_targets...)


            input_data_test = hcat(x1_test_data,x2_test_data)
            input_data_test = input_data_test'


            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, length(x3_test_targets))


            for t in 1:length(x1_test_data)
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input_data_test[:,t], leak_rate)

                predicted_output = output_weights' * next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end


            # recall_score = cor(x3_test_targets,final_outputs').^2
            # recall_score = recall_score[1,1]

            # recall_score = recall_accuracy_ignore_fixation(vec(final_outputs), vec(x3_test_targets), L)

            recall_score = cor(x3_test_targets,final_outputs').^2
            recall_score = recall_score[1,1]

            push!(recalls1,recall_score)
            push!(total_performances, recall_score)

            nrmse_value = compute_rec_nrmse(x3_test_targets, final_outputs',L)
            push!(total_errors,nrmse_value)
        end
        push!(recall_scores, mean(recalls1))
    end
    return total_performances, total_errors
  
end

function res_performance_recall(reservoirs, num_simulation_steps, input_scaling, reg, leak, Ls; threshold=0.8)
    recalls = []  # store best L per reservoir
    corr_vals = []

    @progress for res in reservoirs
        rr = []
        pc1 = []
        
        for kk in 1:num_simulation_steps
            best_L = NaN
            pc = []
            for L in Ls
                pf, _ = res_performance_recall_L([res], num_simulation_steps, input_scaling, reg, leak, L)
                perf = mean(pf)
                push!(pc, perf)

                if perf < threshold
                    break  # stop increasing L once performance drops
                end
                best_L = L
            end
            push!(rr, best_L)
            push!(pc1, pc)
        end
        push!(recalls, rr)
        push!(corr_vals, pc1)
    end

    return recalls, corr_vals
end

function res_performance_recall_L(reservoirs, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate, L)


    #parameters
    res_size = size(reservoirs[1])[1]

    input_dim = 2
    output_dim = 1
    n_trials_train = 100  # Number of training trials
    n_trials_test = 20  # Number of testing trials
    
    recall_scores = []
    total_performances = []
    total_errors = []


    @progress for reservoir in reservoirs
        recalls1 = []

        for i in 1:num_simulation_steps

            # Create weights
            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)

            

            x1_train_data = []
            x2_train_data = []
            x3_train_targets = []

            for x in 1:n_trials_train
                x1,x2,x3 = create_data(L)
                push!(x1_train_data, x1)
                push!(x2_train_data, x2)
                push!(x3_train_targets,x3)
            end

            x1_train_data = vcat(x1_train_data...)
            x2_train_data = vcat(x2_train_data...)
            x3_train_targets = vcat(x3_train_targets...)


            input_data = hcat(x1_train_data,x2_train_data)
            input_data = input_data'

            input_weights*input_data[:,100]

            reservoir_state_trains = zeros(res_size,length(x1_train_data))
            current_state = zeros(res_size) 

            for t in 1:length(x1_train_data)
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input_data[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, x3_train_targets', regularization_coefficient)
            # training_output = output_weights' * reservoir_state_trains

            x1_test_data = []
            x2_test_data = []
            x3_test_targets = []

            for x in 1:n_trials_test
                x1,x2,x3 = create_data(L)
                push!(x1_test_data, x1)
                push!(x2_test_data, x2)
                push!(x3_test_targets,x3)
            end

            x1_test_data = vcat(x1_test_data...)
            x2_test_data = vcat(x2_test_data...)
            x3_test_targets = vcat(x3_test_targets...)


            input_data_test = hcat(x1_test_data,x2_test_data)
            input_data_test = input_data_test'


            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, length(x3_test_targets))


            for t in 1:length(x1_test_data)
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input_data_test[:,t], leak_rate)

                predicted_output = output_weights' * next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end


            # recall_score = cor(x3_test_targets,final_outputs').^2
            # recall_score = recall_score[1,1]

            # recall_score = recall_accuracy_ignore_fixation(vec(final_outputs), vec(x3_test_targets), L)

            recall_score = cor(x3_test_targets,final_outputs').^2
            recall_score = recall_score[1,1]

            push!(recalls1,recall_score)
            push!(total_performances, recall_score)

            nrmse_value = compute_rec_nrmse(x3_test_targets, final_outputs',L)
            push!(total_errors,nrmse_value)
        end
        push!(recall_scores, mean(recalls1))
    end
    return total_performances, total_errors
  
end


function res_performance_decisionmaking(reservoirs, num_simulation_steps, input_scaling, reg, leak, biases; threshold=0.8)
    dms = []  # store best score per reservoir

    x_biases = 1.0 .- biases

    for kk in 1:num_simulation_steps
        best_bias = NaN
        for (id,bias) in enumerate(biases)
            pf, _ = res_performance_decisionmaking_L(reservoirs, num_simulation_steps, bias, input_scaling, reg, leak)
            perf = mean(vcat(pf...))

            if perf < threshold
                break  # stop increasing once performance drops
            end
            best_bias = x_biases[id]
        end

        push!(dms, best_bias)
    end

    return dms
end

function res_performance_decisionmaking_L(reservoirs, num_simulation_steps, bias ,input_scaling, regularization_coefficient, leak_rate)
    num_samples_train = 2000   # Number of training samples
    num_samples_test = 1000   # Number of testing samples
    switch_interval = 100      # Check for a possible switch
    input_dim = 2
    output_dim = 1
    res_size = size(reservoirs[1])[2]
    dm_scores = []
    total_performances = []
    total_errors = []

    @progress for reservoir in reservoirs

        dec_scores = []

        # for i in 1:num_simulation_steps

            # Generate training data
            train_data, train_targets = Main.Reservoirs_src.generate_decisionmaking_data(num_samples_train, switch_interval, bias)
            test_data, test_targets = Main.Reservoirs_src.generate_decisionmaking_data(num_samples_test, switch_interval, bias)

            # Reservoir and input/output layers
            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)


            reservoir_state_trains = zeros(res_size,size(train_data)[2])
            current_state = zeros(res_size) 

            for t in 1:size(train_data)[2]
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, train_data[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, train_targets', regularization_coefficient)

            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, size(test_data)[2])


            for t in 1:size(test_data)[2]
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_data[:,t], leak_rate)

                # println("Next state: ", size(next_state))
                # println("Output weights: ", size(output_weights))
                predicted_output = output_weights' * next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            # dm_score = cor(test_targets,final_outputs').^2
            # dm_score = dm_score[1,1]

            dm_score = compute_dm_accuracy(final_outputs, test_targets', threshold=0.0)
            # dm_score = compute_dm_accuracy_window(final_outputs, test_targets', decision_window=switch_interval, threshold=0.0)

            push!(dec_scores, dm_score)
            push!(total_performances, dm_score)

            error_dm = compute_dm_nrmse(test_targets, final_outputs')

            # error_rate = compute_dm_error_rate(test_targets, final_outputs')
            push!(total_errors, error_dm)

        # end

        push!(dm_scores,dec_scores)
    end

    return total_performances, total_errors

end

function res_performance_delay_decisionmaking_old(reservoirs, num_simulation_steps, variances, input_scaling, regularization_coefficient, leak_rate)

    # Parameters
    num_trials = 100         # Number of trials
    stim_duration = 10       # Duration of stimulus presentation (time steps)
    resp_duration = 10       # Duration of response period (time steps)


    ddm_scores = []
    total_performances = []
    total_errors = []

    input_dim = 2
    output_dim = 1

    res_size = size(reservoirs[1],2)

    for reservoir in reservoirs 

        dec_scores = []

        for i in 1:num_simulation_steps
            variance = variances[i]
            # Generate data
            training_inputs, training_target_output = Main.Reservoirs_src.generate_delay_decisionmaking_task_data(num_trials, stim_duration, resp_duration,variance)
            # Generate test data
            test_inputs, test_target_output = Main.Reservoirs_src.generate_delay_decisionmaking_task_data(10, stim_duration, resp_duration,variance)
            #

            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)

            reservoir_state_trains = zeros(res_size,size(training_inputs)[2])
            current_state = zeros(res_size) 

            for t in 1:size(training_inputs)[2]
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, training_inputs[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end


            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, training_target_output', regularization_coefficient)
            training_output = output_weights' * reservoir_state_trains


            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, size(test_inputs)[2])


            for t in 1:size(test_inputs)[2]
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_inputs[:,t], leak_rate)

                predicted_output = output_weights' * next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            trial_len = stim_duration + resp_duration  # or whatever full trial length is
            fix_len = stim_duration  # fixation/stimulus period to ignore
            ddm_score = delay_decision_accuracy_ignore_fixation(final_outputs[:], test_target_output, trial_len, fix_len)


            push!(dec_scores, ddm_score)
            push!(total_performances, ddm_score)
            # error_rate_filtered = compute_error_rate_ignore_fixation(test_target_output, final_outputs')

            error_rate_filtered = ddm_nrmse_ignore_fixation(vec(test_target_output), vec(final_outputs'), trial_len, fix_len)


            push!(total_errors, error_rate_filtered)
        end

        push!(ddm_scores, mean(dec_scores))

    end

    return total_performances, total_errors

end

function res_performance_delay_decisionmaking(reservoirs, num_simulation_steps, input_scaling, reg, leak, variances; threshold=0.8)
    ddms = []  # store best L per reservoir


    @progress for res in reservoirs
        rr = []
        for kk in 1:num_simulation_steps
            best_var = NaN
            for var in variances
                pf, _ = res_performance_delay_decisionmaking_L([res], num_simulation_steps, var, input_scaling, reg, leak)
                perf = mean(pf)

                if perf < threshold
                    break  # stop increasing L once performance drops
                end
                best_var = var
            end
            push!(rr, best_var)
        end
        push!(ddms, rr)
    end

    return ddms
end


function res_performance_delay_decisionmaking_L(reservoirs, num_simulation_steps, var, input_scaling, regularization_coefficient, leak_rate)

    # Parameters
    num_trials = 100         # Number of trials
    stim_duration = 20       # Duration of stimulus presentation (time steps)
    resp_duration = 10       # Duration of response period (time steps)


    ddm_scores = []
    total_performances = []
    total_errors = []

    input_dim = 2
    output_dim = 1

    res_size = size(reservoirs[1],2)

    for reservoir in reservoirs 

        dec_scores = []

        for i in 1:num_simulation_steps
            variance = var
            # Generate data
            training_inputs, training_target_output = Main.Reservoirs_src.generate_delay_decisionmaking_task_data(num_trials, stim_duration, resp_duration,variance)
            # Generate test data
            test_inputs, test_target_output = Main.Reservoirs_src.generate_delay_decisionmaking_task_data(10, stim_duration, resp_duration,variance)
            #

            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)

            reservoir_state_trains = zeros(res_size,size(training_inputs)[2])
            current_state = zeros(res_size) 

            for t in 1:size(training_inputs)[2]
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, training_inputs[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end


            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, training_target_output', regularization_coefficient)
            training_output = output_weights' * reservoir_state_trains


            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, size(test_inputs)[2])


            for t in 1:size(test_inputs)[2]
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_inputs[:,t], leak_rate)

                predicted_output = output_weights' * next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end


            # ddm_score = cor(test_target_output,final_outputs').^2
            # ddm_score = ddm_score[1,1]

            trial_len = stim_duration + resp_duration  # or whatever full trial length is
            fix_len = stim_duration  # fixation/stimulus period to ignore
            ddm_score = delay_decision_accuracy_ignore_fixation(final_outputs[:], test_target_output, trial_len, fix_len)


            push!(dec_scores, ddm_score)
            push!(total_performances, ddm_score)
            # error_rate_filtered = compute_error_rate_ignore_fixation(test_target_output, final_outputs')

            error_rate_filtered = ddm_nrmse_ignore_fixation(vec(test_target_output), vec(final_outputs'), trial_len, fix_len)


            push!(total_errors, error_rate_filtered)
        end

        push!(ddm_scores, mean(dec_scores))

    end

    return total_performances, total_errors

end

function res_performance_oscillator(reservoirs, fulldata, threshold, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate)

    total_performances = []
    total_errors = []
    savetimestep = 0.1
    train_period_length = 1000
    test_period_length = 2000

    train_data = fulldata[1:train_period_length]


    @progress for reservoir in reservoirs

        input_size = 1
        output_size = 1 
        reservoir_size = size(reservoir)[1]

        # trying different setups of input and output layers
        for i in 1:num_simulation_steps
            max_start = length(fulldata) - train_period_length - test_period_length + 1
            rr = rand(1:max_start)
            test_start = train_period_length + rr
            test_end = test_start + test_period_length - 1
            test_data = fulldata[test_start:test_end]

            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)
            
            # Initialize reservoir states
            reservoir_states = zeros(reservoir_size, length(train_data)-1)
            current_state = zeros(reservoir_size)
            
            # Training
            for i in 1:length(train_data)-1
                input = train_data[i]  # Input must be a vector
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
                reservoir_states[:, i] = current_state
            end
            
            # Train output weights
            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, train_data[2:end]', regularization_coefficient)

            # Initialize predictions
            predictions = zeros(output_size, test_period_length)
            last_state = reservoir_states[:, end]
            
            # Predict using only previous predictions after 100 steps
            for i in 1:test_period_length
                if i < 100
                    input = test_data[i]
                else
                    input = predictions[i-1]
                end
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
                xx = output_weights' * next_state
                predictions[i] = xx[1]
                last_state = next_state
            end
            
            pp = predictions[1:end-1]
            ppp = test_data[2:end]
            
            t1, measure = Main.Reservoirs_src.valid_time_oscillator(threshold, reshape(pp, 1, length(pp)), reshape(ppp, 1, length(ppp)), savetimestep)
        
            push!(total_performances, measure)

            pp1 = predictions[100:end-1]
            ppp1 = test_data[101:end]

            nrmse_val = compute_nrmse_oscillator(ppp1, pp1)

            push!(total_errors, nrmse_val)

        end

    end

    return total_performances, total_errors

end

function res_performance_lotka(reservoirs, fulldata, threshold, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate)

    total_performances = []
    total_errors = []
    dt = 0.1
    train_period_length = 1000
    test_period_length = 2000

    train_data = fulldata[1:train_period_length]



    @progress for reservoir in reservoirs

        input_size = 1
        output_size = 1 
        reservoir_size = size(reservoir)[1]

        # trying different setups of input and output layers
        for i in 1:num_simulation_steps
            max_start = length(fulldata) - train_period_length - test_period_length + 1
            rr = rand(1:max_start)
            test_start = train_period_length + rr
            test_end = test_start + test_period_length - 1
            test_data = fulldata[test_start:test_end]

            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)
            
            # Initialize reservoir states
            reservoir_states = zeros(reservoir_size, length(train_data)-1)
            current_state = zeros(reservoir_size)
            
            # Training
            for i in 1:length(train_data)-1
                input = train_data[i]  # Input must be a vector
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
                reservoir_states[:, i] = current_state
            end
            
            # Train output weights
            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, train_data[2:end]', regularization_coefficient)

            # Initialize predictions
            predictions = zeros(output_size, test_period_length)
            last_state = reservoir_states[:, end]
            
            # Predict using only previous predictions after 100 steps
            for i in 1:test_period_length
                if i < 100
                    input = test_data[i]
                else
                    input = predictions[i-1]
                end
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
                xx = output_weights' * next_state
                predictions[i] = xx[1]
                last_state = next_state
            end
            
            pp = predictions[1:end-1]
            ppp = test_data[2:end]
            
            t1, measure = Main.Reservoirs_src.valid_time_lotka(threshold, reshape(pp, 1, length(pp)), reshape(ppp, 1, length(ppp)), dt)

            push!(total_performances, measure)

            pp1 = predictions[100:end-1]
            ppp1 = test_data[101:end]

            nrmse_val = compute_nrmse_lotka(reshape(ppp1,1,length(ppp1)), reshape(pp1,1,length(pp1)))

            
            push!(total_errors, nrmse_val)

        end

    end

    return total_performances, total_errors

end

function res_performance_lotka_2d(reservoirs, fulldata, threshold, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate)
    total_performances = []
    total_errors = []
    dt = 0.1
    train_period_length = 1000
    test_period_length = 2000

    train_data = fulldata[1:train_period_length,:]'

    @progress for reservoir in reservoirs
        input_size = 2
        output_size = 2
        reservoir_size = size(reservoir, 1)

        for sim in 1:num_simulation_steps
            max_start = size(fulldata, 1) - train_period_length - test_period_length + 1
            rr = rand(1:max_start)
            test_start = train_period_length + rr
            test_end = test_start + test_period_length - 1
            test_data = fulldata[test_start:test_end,:]'

            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            # Initialize reservoir states
            reservoir_states = zeros(reservoir_size, size(train_data, 2)-1)
            current_state = zeros(reservoir_size)

            # Training
            for i in 1:size(train_data, 2)-1
                input = train_data[:, i]
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
                reservoir_states[:, i] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, train_data[:, 2:end], regularization_coefficient)

            # Prediction
            predictions = zeros(output_size, test_period_length)
            last_state = reservoir_states[:, end]

            for i in 1:test_period_length
                input = i < 100 ? test_data[:, i] : predictions[:, i-1]
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
                predictions[:, i] = output_weights' * next_state
                last_state = next_state
            end

            pp = predictions[:, 1:end-1]
            ppp = test_data[:, 2:end]

            t1, measure = Main.Reservoirs_src.valid_time_lotka(threshold, pp, ppp, dt)
            push!(total_performances, measure)

            pp1 = predictions[:, 100:end-1]
            ppp1 = test_data[:, 101:end]
            nrmse_val = compute_nrmse_lotka(ppp1, pp1)

            push!(total_errors, nrmse_val)
        end
    end

    return total_performances, total_errors
end

function res_performance_lorenz(reservoirs, train_data, test_data, num_simulation_steps, threshhold, input_scaling, regularization_coefficient, leak_rate)
    res_performances = []
    total_performances = []
    total_errors = []
    lorenz_savetimestep = 0.02
    train_period_length= 2000
    test_period_length = 1000

    ross_timestep = 0.05

    @progress for reservoir in reservoirs
        #parameters
        input_size = 3
        output_size = 3 
        reservoir_size = size(reservoir,2)

        traindata = train_data[:,1:end-1]
        targetdata = train_data[:,2:end]


        # trying different setups of input and output layers
        for i in 1:num_simulation_steps
            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            reservoir_states = zeros(reservoir_size, size(traindata,2))

            # Initialising the reservoir
            current_state = zeros(reservoir_size)   


            # Training the output layer
            for i in 1:size(traindata, 2)
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, traindata[:, i], leak_rate)
                reservoir_states[:, i] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, targetdata, regularization_coefficient)

            measures = []

            testid = rand(1:6)
            testdata = test_data[testid]
            # Initialising predictions
            predictions = zeros(output_size, test_period_length)

            # Starting with the last state from training
            last_state = reservoir_states[:, end]

            for i in 1:test_period_length
                # Generate the next state
                if i < 100
                    input = testdata[:,i]
                else
                    input = predictions[:, i-1]
                end
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
                predicted_output = output_weights' * next_state
            
                # Storing the predicted output
                predictions[:, i] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            savetimestep=0.02
            # println(size(testdata,2))
            
            t1, measure = Main.Reservoirs_src.valid_time_lorenz(threshhold, predictions, testdata[:,1:end-1], lorenz_savetimestep)

            edit_test_data = testdata[:,99:999]
            edit_predictions = predictions[:,100:end]


            push!(measures, measure)
            push!(total_performances, measure)

            nrmse_value = compute_nrmse_lorenz(edit_test_data, edit_predictions)

            push!(total_errors, nrmse_value)

            # push!(measurements, mean(measures))

        end

        # push!(res_performances, mean(measurements))

    end

    return total_performances, total_errors
                            
end

function res_performance_rossler(reservoirs, train_data, test_data, num_simulation_steps, threshhold, input_scaling, regularization_coefficient, leak_rate)

    res_performances = []
    total_performances = []
    total_errors = []
    rossler_savetimestep = 0.05
    train_period_length= 10000
    test_period_length = 5000

    @progress for reservoir in reservoirs

        input_size = 3
        output_size = 3 
        reservoir_size = size(reservoir)[1]

        traindata = train_data[:,1:end-1]
        targetdata = train_data[:,2:end]

        measurements = []

        # trying different setups of input and output layers
        for i in 1:num_simulation_steps

            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size);
            reservoir_states = zeros(reservoir_size, size(traindata)[2])

            # Initialising the reservoir
            current_state = zeros(reservoir_size)   


            # Training the output layer
            for i in 1:size(traindata, 2)-1
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, traindata[:, i], leak_rate)
                reservoir_states[:, i] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, targetdata, regularization_coefficient)
            
            
            measures = []
            
            testid = rand(1:6)
            testdata = test_data[testid][:,1:end-1]
            # Initialising predictions
            predictions = zeros(output_size, test_period_length)

            # Starting with the last state from training
            last_state = reservoir_states[:, end]

            for i in 1:test_period_length
                # Generate the next state
                if i < 100
                    input = testdata[:,i]
                else
                    input = predictions[:, i-1]
                end
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
                predicted_output = output_weights' * next_state
                # Storing the predicted output
                predictions[:, i] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            # measure of performance
            t1, measure = Main.Reservoirs_src.valid_time_rossler(threshhold, predictions, testdata[:,1:end], rossler_savetimestep)

            

            push!(measures, measure)
            push!(total_performances, measure)

            edit_test_data = testdata[:,99:end-1]
            edit_predictions = predictions[:,100:end]

            # println("size of edit testdata: ", size(edit_test_data))
            # println("size of edit predictions: ", size(edit_predictions))

            nrmse_val = compute_nrmse_rossler(edit_test_data,edit_predictions)
            push!(total_errors, nrmse_val)
 

        end


    end

    return total_performances, total_errors

end



###########
###########
###########

# NB these are old pruning functions

function memory_pruning(reservoirs, node_ids, num_simulation_steps, input_scaling, spectral_radius, regularization_coefficient, leak_rate)

    reservoirs1 = Main.ConnectomeFunctions.scale_matrices(deepcopy(reservoirs),spectral_radius)
    #parameters
    reservoir_size = size(reservoirs[1])[1]
    # input_scaling = 0.1
    input_size = 1
    output_size = 100
    train_length = 4000
    test_length = 1000

    total_performances = []
    total_criticalities = []

    res_PRs = []
    res_sparsities = []
    res_SRs = []
    activity_PRs = []


    @progress for (id,reservoir) in enumerate(reservoirs1)


        for i in 1:num_simulation_steps


            # Create input and output weights
            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            for node_id in node_ids[id]
                input_weights[node_id,:] .= 0
                reservoir[node_id,:] .= 0
                reservoir[:,node_id] .= 0
            end

            res_eigenvalues = eigvals(Matrix(reservoir))
            push!(res_PRs, Main.Reservoirs_src.participation_ratio(abs.(res_eigenvalues)))

            res_sparsity = Main.ConnectomeFunctions.compute_sparsity(reservoir)
            push!(res_sparsities, res_sparsity)

            res_SR = maximum(abs.(res_eigenvalues))
            push!(res_SRs, res_SR)


            # Generate random input sequence
            X_train = rand(train_length) .- 0.5
            X_test = rand(test_length) .- 0.5

            # Generate target data for training the output layer
            target_data = Main.Reservoirs_src.generate_memory_target_data(X_train, output_size)'


            # Update reservoir state for training
            reservoir_state_train = zeros(reservoir_size, train_length)

            current_state = zeros(reservoir_size)   

            for t in 1:train_length
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, X_train[t], leak_rate)
                reservoir_state_train[:, t] = current_state
            end

            # Train output weights
            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_train, target_data, regularization_coefficient)

            final_outputs = zeros(output_size, test_length)


            # Starting with the last state from training
            last_state = reservoir_state_train[:, end]

            reservoir_internal_states = zeros(reservoir_size,test_length)

            for t in 1:test_length
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, X_test[t], leak_rate)

                predicted_output = output_weights' * next_state

                reservoir_internal_states[:,t] = next_state
                # Storing the predicted output
                final_outputs[:, t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            _, ev, _ = Main.Reservoirs_src.pca(reservoir_internal_states)

            push!(activity_PRs, Main.Reservoirs_src.participation_ratio(abs.(ev)))

            laststate = reservoir_internal_states[:,end]

            memcriticality = max_lyapunov_exponent(laststate, 1000, reservoir, input_weights, X_test'; ε=1e-7)

            push!(total_criticalities, memcriticality)


            X = vcat([X_train, X_test]...)
            x_output = Main.Reservoirs_src.generate_memory_target_data(X, output_size)'
            test_output = x_output[:,4001:end]


            memory_capacity = 0
            # Working out measure of memory capacity
            for j in 1:output_size
                # Calculate squared Pearson correlation coefficient
                rho = cor(test_output[j,:], final_outputs[j,:])^2
                # Accumulate MC score
                memory_capacity += rho
            end

            push!(total_performances, memory_capacity)

        end

    end

    return total_performances, res_PRs, res_sparsities, res_SRs, activity_PRs, total_criticalities

                    
end

function recall_pruning(reservoirs, node_ids, num_simulation_steps, input_scaling, spectral_radius, regularization_coefficient, leak_rate)

    reservoirs = Main.ConnectomeFunctions.scale_matrices(deepcopy(reservoirs),spectral_radius)
    #parameters
    res_size = size(reservoirs[1])[1]

    input_dim = 2
    output_dim = 1
    n_trials_train = 200  # Number of training trials
    n_trials_test = 50  # Number of testing trials


    total_performances = []
    total_criticalities = []
    res_PRs = []
    res_sparsities = []
    res_SRs = []
    activity_PRs = []


    @progress for (id,reservoir) in enumerate(reservoirs)


        for i in 1:num_simulation_steps

            L = 40
            # Create weights
            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)

            for node_id in node_ids[id]
                input_weights[node_id,:] .= 0
                reservoir[node_id,:] .= 0
                reservoir[:,node_id] .= 0
            end

            res_eigenvalues = eigvals(Matrix(reservoir))
            push!(res_PRs, Main.Reservoirs_src.participation_ratio(abs.(res_eigenvalues)))

            res_sparsity = Main.ConnectomeFunctions.compute_sparsity(reservoir)
            push!(res_sparsities, res_sparsity)

            res_SR = maximum(abs.(res_eigenvalues))
            push!(res_SRs, res_SR)

            

            x1_train_data = []
            x2_train_data = []
            x3_train_targets = []

            for x in 1:n_trials_train
                x1,x2,x3 = create_data(L)
                push!(x1_train_data, x1)
                push!(x2_train_data, x2)
                push!(x3_train_targets,x3)
            end

            x1_train_data = vcat(x1_train_data...)
            x2_train_data = vcat(x2_train_data...)
            x3_train_targets = vcat(x3_train_targets...)


            input_data = hcat(x1_train_data,x2_train_data)
            input_data = input_data'

            input_weights*input_data[:,100]

            reservoir_state_trains = zeros(res_size,length(x1_train_data))
            current_state = zeros(res_size) 

            for t in 1:length(x1_train_data)
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input_data[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, x3_train_targets', regularization_coefficient)
            # training_output = output_weights' * reservoir_state_trains

            x1_test_data = []
            x2_test_data = []
            x3_test_targets = []

            for x in 1:n_trials_test
                x1,x2,x3 = create_data(L)
                push!(x1_test_data, x1)
                push!(x2_test_data, x2)
                push!(x3_test_targets,x3)
            end

            x1_test_data = vcat(x1_test_data...)
            x2_test_data = vcat(x2_test_data...)
            x3_test_targets = vcat(x3_test_targets...)


            input_data_test = hcat(x1_test_data,x2_test_data)
            input_data_test = input_data_test'


            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, length(x3_test_targets))
            reservoir_internal_states = zeros(res_size,length(x1_test_data))

            for t in 1:length(x1_test_data)
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input_data_test[:,t], leak_rate)

                predicted_output = output_weights' * next_state

                reservoir_internal_states[:,t] = next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            _, ev, _ = Main.Reservoirs_src.pca(reservoir_internal_states)

            push!(activity_PRs, Main.Reservoirs_src.participation_ratio(abs.(ev)))

            laststate = reservoir_internal_states[:,end]

            reccriticality = max_lyapunov_exponent(laststate, length(x1_test_data), reservoir, input_weights, input_data_test; ε=1e-7)

            push!(total_criticalities, reccriticality)

            # recall_score = recall_accuracy_ignore_fixation(vec(final_outputs),vec(x3_test_targets), L)
 
            recall_score = cor(x3_test_targets,final_outputs').^2
            recall_score = recall_score[1,1]

            push!(total_performances, recall_score)
        end
    end
    return total_performances, res_PRs, res_sparsities, res_SRs, activity_PRs, total_criticalities
end

function decisionmaking_pruning(reservoirs, node_ids, num_simulation_steps, biases, input_scaling, spectral_radius, regularization_coefficient, leak_rate)

    reservoirs1 = Main.ConnectomeFunctions.scale_matrices(deepcopy(reservoirs),spectral_radius)
    num_samples_train = 1000   # Number of training samples
    num_samples_test = 20000   # Number of testing samples
    switch_interval = 100      # Check for a possible switch every 100 timesteps
    input_dim = 2
    output_dim = 1
    res_size = size(reservoirs[1])[2]


  
    total_performances = []
    total_criticalities = []
    res_PRs = []
    res_sparsities = []
    res_SRs = []
    activity_PRs = []
 
    # eigenvalues = []

    @progress for (id,reservoir) in enumerate(reservoirs1)


        for i in 1:num_simulation_steps
            # Generate training data
            bias = biases[i]
            train_data, train_targets = Main.Reservoirs_src.generate_decisionmaking_data(num_samples_train, switch_interval, bias)
            test_data, test_targets = Main.Reservoirs_src.generate_decisionmaking_data(num_samples_test, switch_interval, bias)

            # Reservoir and input/output layers
            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)

            for node_id in node_ids[id]
                input_weights[node_id,:] .= 0
                reservoir[node_id,:] .= 0
                reservoir[:,node_id] .= 0
            end

            # res_eigenvalues = eigvals(Matrix(reservoir))
            # push!(res_PRs, Main.Reservoirs_src.participation_ratio(abs.(res_eigenvalues)))
            # push!(evals, res_eigenvalues)

            # res_sparsity = Main.ConnectomeFunctions.compute_sparsity(reservoir)
            # push!(res_sparsities, res_sparsity)

            # res_SR = maximum(abs.(res_eigenvalues))
            # push!(res_SRs, res_SR)


            reservoir_state_trains = zeros(res_size,size(train_data)[2])
            current_state = zeros(res_size) 

            for t in 1:size(train_data)[2]
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, train_data[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, train_targets', regularization_coefficient)

            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, size(test_data)[2])
            reservoir_internal_states = zeros(res_size, size(test_data)[2])

            for t in 1:size(test_data)[2]
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_data[:,t], leak_rate)

                predicted_output = output_weights' * next_state
                reservoir_internal_states[:,t] = next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            _, ev, _ = Main.Reservoirs_src.pca(reservoir_internal_states)

            push!(activity_PRs, Main.Reservoirs_src.participation_ratio(abs.(ev)))

            laststate = reservoir_internal_states[:,end]

            dmcriticality = max_lyapunov_exponent(laststate, size(test_data)[2], reservoir, input_weights, test_data; ε=1e-7)

            push!(total_criticalities, dmcriticality)

            dm_score = compute_dm_accuracy(final_outputs, test_targets', threshold=0.0)
            # dm_score = interval_accuracy1(final_outputs, test_targets', switch_interval)

            # dm_score = cor(test_targets,final_outputs').^2
            # dm_score = dm_score[1,1]


            push!(total_performances, dm_score)

        end


        # push!(eigenvalues, evals)
    end

    return total_performances, res_PRs, res_sparsities, res_SRs, activity_PRs, total_criticalities

end

function delay_decisionmaking_pruning(reservoirs1, node_ids, num_simulation_steps, variances, input_scaling, spectral_radius, regularization_coefficient, leak_rate)

    reservoirs = Main.ConnectomeFunctions.scale_matrices(deepcopy(reservoirs1),spectral_radius)
    # Parameters
    num_trials = 100         # Number of trials
    stim_duration = 10       # Duration of stimulus presentation (time steps)
    resp_duration = 10       # Duration of response period (time steps)

   
    total_performances = []
    total_criticalities = []
    res_PRs = []
    res_sparsities = []
    res_SRs = []
    activity_PRs = []

    input_dim = 2
    output_dim = 1

    res_size = size(reservoirs[1],2)
    
    for (id,reservoir) in enumerate(reservoirs)

        for i in 1:num_simulation_steps
            variance = variances[i]

            # Generate data
            training_inputs, training_target_output = Main.Reservoirs_src.generate_delay_decisionmaking_task_data(num_trials, stim_duration, resp_duration,variance)
            # Generate test data
            test_inputs, test_target_output = Main.Reservoirs_src.generate_delay_decisionmaking_task_data(100, stim_duration, resp_duration,variance)
            #

            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)

            for node_id in node_ids[id]
                input_weights[node_id,:] .= 0
                reservoir[node_id,:] .= 0
                reservoir[:,node_id] .= 0
            end

            res_eigenvalues = eigvals(Matrix(reservoir))
            push!(res_PRs, Main.Reservoirs_src.participation_ratio(abs.(res_eigenvalues)))

            res_sparsity = Main.ConnectomeFunctions.compute_sparsity(reservoir)
            push!(res_sparsities, res_sparsity)

            res_SR = maximum(abs.(res_eigenvalues))
            push!(res_SRs, res_SR)

            reservoir_state_trains = zeros(res_size,size(training_inputs)[2])
            current_state = zeros(res_size) 

            for t in 1:size(training_inputs)[2]
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, training_inputs[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end


            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, training_target_output', regularization_coefficient)
            training_output = output_weights' * reservoir_state_trains


            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, size(test_inputs)[2])

            reservoir_internal_states = zeros(res_size, size(test_inputs)[2])

            for t in 1:size(test_inputs)[2]
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_inputs[:,t], leak_rate)

                predicted_output = output_weights' * next_state
                reservoir_internal_states[:,t] = next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            _, ev, _ = Main.Reservoirs_src.pca(reservoir_internal_states)

            push!(activity_PRs, Main.Reservoirs_src.participation_ratio(abs.(ev)))

            laststate = reservoir_internal_states[:,end]

            ddmcriticality = max_lyapunov_exponent(laststate, size(test_inputs)[2], reservoir, input_weights, test_inputs; ε=1e-7)

            push!(total_criticalities, ddmcriticality)


            trial_len = stim_duration + resp_duration  # or whatever full trial length is
            fix_len = stim_duration  # fixation/stimulus period to ignore
            ddm_score = delay_decision_accuracy_ignore_fixation(final_outputs[:], test_target_output, trial_len, fix_len)

            # ddm_score = cor(test_target_output,final_outputs').^2
            # ddm_score = ddm_score[1,1]

            push!(total_performances, ddm_score)
        end


    end

    return total_performances, res_PRs, res_sparsities, res_SRs, activity_PRs, total_criticalities

end

function osc_pruning(reservoirs, node_ids, osc_data, num_simulation_steps, threshold, input_scaling, spectral_radius, regularization_coefficient, leak_rate)
    reservoirs = Main.ConnectomeFunctions.scale_matrices(deepcopy(reservoirs),spectral_radius)

    savetimestep = 0.1
    train_period_length = 1000
    test_period_length = 5000

    traindata = osc_data[1:train_period_length]

    total_performances = []
    total_criticalities = []
    res_PRs = []
    res_sparsities = []
    res_SRs = []
    activity_PRs = []


    @progress for (id,reservoir) in enumerate(reservoirs)

        #parameters
        input_size = 1
        output_size = 1
        reservoir_size = size(reservoir,2)


        # trying different setups of input and output layers
        for ii in 1:num_simulation_steps

            max_start = length(osc_data) - train_period_length - test_period_length + 1
            rr = rand(1:max_start)
            test_start = train_period_length + rr
            test_end = test_start + test_period_length - 1
            testdata = osc_data[test_start:test_end]


            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            for node_id in node_ids[id]
                input_weights[node_id,:] .= 0
                reservoir[node_id,:] .= 0
                reservoir[:,node_id] .= 0
            end

            res_eigenvalues = eigvals(Matrix(reservoir))
            push!(res_PRs, Main.Reservoirs_src.participation_ratio(abs.(res_eigenvalues)))

            res_sparsity = Main.ConnectomeFunctions.compute_sparsity(reservoir)
            push!(res_sparsities, res_sparsity)

            res_SR = maximum(abs.(res_eigenvalues))
            push!(res_SRs, res_SR)

            # Initialize reservoir states
            reservoir_states = zeros(reservoir_size, length(traindata)-1)
            current_state = zeros(reservoir_size)
            
            # Training
            for i in 1:length(traindata)-1
                input = traindata[i]  # Input must be a vector
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
                reservoir_states[:, i] = current_state
            end


            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, traindata[2:end]', regularization_coefficient)

            # Initialize predictions
            predictions = zeros(output_size, test_period_length)
            reservoir_internal_states = zeros(reservoir_size, test_period_length)
            last_state = reservoir_states[:, end]
            
            # Predict using only previous predictions after 100 steps
            for i in 1:test_period_length
                if i < 100
                    input = testdata[i]
                else
                    input = predictions[i-1]
                end
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
                reservoir_internal_states[:,i] = next_state
                xx = output_weights' * next_state
                predictions[i] = xx[1]
                last_state = next_state
            end
            
            pp = predictions[1:end-1]
            ppp = testdata[2:end]
            
            t1, measure = Main.Reservoirs_src.valid_time_oscillator(threshold, reshape(pp, 1, length(pp)), reshape(ppp, 1, length(ppp)), savetimestep)
        
            push!(total_performances, measure)

            _, ev, _ = Main.Reservoirs_src.pca(reservoir_internal_states)

            push!(activity_PRs, Main.Reservoirs_src.participation_ratio(abs.(ev)))

            laststate = reservoir_internal_states[:,end]

            osccriticality = max_lyapunov_exponent(laststate, length(testdata), reservoir, input_weights, testdata'; ε=1e-7)

            push!(total_criticalities, osccriticality)
    
        end

    end

    return total_performances, res_PRs, res_sparsities, res_SRs, activity_PRs, total_criticalities
         
end

function lot_pruning(reservoirs, node_ids, lot_data, num_simulation_steps, threshold, input_scaling, spectral_radius, regularization_coefficient, leak_rate)
    reservoirs = Main.ConnectomeFunctions.scale_matrices(deepcopy(reservoirs),spectral_radius)

    dt = 0.01
    train_period_length = 1000
    test_period_length = 2000

    traindata = lot_data[1:train_period_length,:]'

    total_performances = []
    total_criticalities = []
    res_PRs = []
    res_sparsities = []
    res_SRs = []
    activity_PRs = []


    @progress for (id,reservoir) in enumerate(reservoirs)

        #parameters
        input_size = 2
        output_size = 2
        reservoir_size = size(reservoir,2)


        # trying different setups of input and output layers
        for ii in 1:num_simulation_steps

            max_start = size(lot_data, 1) - train_period_length - test_period_length + 1
            rr = rand(1:max_start)
            test_start = train_period_length + rr
            test_end = test_start + test_period_length - 1
            testdata = lot_data[test_start:test_end,:]'


            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            for node_id in node_ids[id]
                input_weights[node_id,:] .= 0
                reservoir[node_id,:] .= 0
                reservoir[:,node_id] .= 0
            end

            res_eigenvalues = eigvals(Matrix(reservoir))
            push!(res_PRs, Main.Reservoirs_src.participation_ratio(abs.(res_eigenvalues)))

            res_sparsity = Main.ConnectomeFunctions.compute_sparsity(reservoir)
            push!(res_sparsities, res_sparsity)

            res_SR = maximum(abs.(res_eigenvalues))
            push!(res_SRs, res_SR)

            # Initialize reservoir states
            reservoir_states = zeros(reservoir_size, size(traindata, 2)-1)
            current_state = zeros(reservoir_size)
            
            # Training
            for i in 1:size(traindata, 2)-1
                input = traindata[:,i]  # Input must be a vector
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
                reservoir_states[:, i] = current_state
            end


            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, traindata[:,2:end], regularization_coefficient)

            # Initialize predictions
            predictions = zeros(output_size, test_period_length)
            reservoir_internal_states = zeros(reservoir_size, test_period_length)
            last_state = reservoir_states[:, end]

    
            
            # Predict using only previous predictions after 100 steps
            for i in 1:test_period_length
                input = i < 100 ? testdata[:, i] : predictions[:, i-1]
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
                predictions[:, i] = output_weights' * next_state
                last_state = next_state
                reservoir_internal_states[:,i] = next_state

            end
            
            pp = predictions[:, 1:end-1]
            ppp = testdata[:, 2:end]

            t1, measure = Main.Reservoirs_src.valid_time_lotka(threshold, pp, ppp, dt)

            push!(total_performances, measure)

            _, ev, _ = Main.Reservoirs_src.pca(reservoir_internal_states)

            push!(activity_PRs, Main.Reservoirs_src.participation_ratio(abs.(ev)))

            laststate = reservoir_internal_states[:,end]

            osccriticality = max_lyapunov_exponent(laststate, size(testdata,1), reservoir, input_weights, testdata; ε=1e-7)

            push!(total_criticalities, osccriticality)
    
        end

    end

    return total_performances, res_PRs, res_sparsities, res_SRs, activity_PRs, total_criticalities
         
end

function lorenz_pruning(reservoirs, node_ids, train_data, test_data, num_simulation_steps, threshhold, input_scaling, spectral_radius, regularization_coefficient, leak_rate)

    reservoirs = Main.ConnectomeFunctions.scale_matrices(deepcopy(reservoirs),spectral_radius)
    total_performances = []
    total_criticalities = []
    res_PRs = []
    res_sparsities = []
    res_SRs = []
    activity_PRs = []


    @progress for (id,reservoir) in enumerate(reservoirs)

        #parameters
        input_size = 3
        output_size = 3 
        reservoir_size = size(reservoir,2)

        traindata = train_data[:,1:end-1]
        targetdata = train_data[:,2:end]

        # trying different setups of input and output layers
        for ii in 1:num_simulation_steps
            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            for node_id in node_ids[id]
                input_weights[node_id,:] .= 0
                reservoir[node_id,:] .= 0
                reservoir[:,node_id] .= 0
            end

            res_eigenvalues = eigvals(Matrix(reservoir))
            push!(res_PRs, Main.Reservoirs_src.participation_ratio(abs.(res_eigenvalues)))

            res_sparsity = Main.ConnectomeFunctions.compute_sparsity(reservoir)
            push!(res_sparsities, res_sparsity)

            res_SR = maximum(abs.(res_eigenvalues))
            push!(res_SRs, res_SR)

            reservoir_states = zeros(reservoir_size, size(traindata,2))

            # Initialising the reservoir
            current_state = zeros(reservoir_size)   


            # Training the output layer
            for i in 1:size(traindata, 2)
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, traindata[:, i], leak_rate)
                reservoir_states[:, i] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, targetdata, regularization_coefficient)

            testid = rand(1:6)
            testdata = test_data[testid]
            # Initialising predictions
            predictions = zeros(output_size, size(testdata,2))

            # Starting with the last state from training
            last_state = reservoir_states[:, end]

            reservoir_internal_states = zeros(reservoir_size,size(testdata,2))

            for i in 1:size(testdata,2)
                # Generate the next state
                if i < 100
                    input = testdata[:,i]
                else
                    input = predictions[:, i-1]
                end
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
                predicted_output = output_weights' * next_state

                reservoir_internal_states[:,i] = next_state
            
                # Storing the predicted output
                predictions[:, i] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            _, ev, _ = Main.Reservoirs_src.pca(reservoir_internal_states)

            push!(activity_PRs, Main.Reservoirs_src.participation_ratio(abs.(ev)))

            laststate = reservoir_internal_states[:,end]

            lorcriticality = max_lyapunov_exponent(laststate, size(testdata)[2], reservoir, input_weights, testdata; ε=1e-7)

            push!(total_criticalities, lorcriticality)

            savetimestep=0.02
            
            t1, measure = Main.Reservoirs_src.valid_time_lorenz(threshhold, predictions, testdata[:,1:end], savetimestep)

            push!(total_performances, measure)
    
        end

    end

    return total_performances, res_PRs, res_sparsities, res_SRs, activity_PRs, total_criticalities
                            
end

function rossler_pruning(reservoirs, node_ids, train_data, test_data, num_simulation_steps, threshhold, input_scaling, spectral_radius, regularization_coefficient, leak_rate)
    reservoirs = Main.ConnectomeFunctions.scale_matrices(deepcopy(reservoirs),spectral_radius)

    total_performances = []
    total_criticalities = []
    res_PRs = []
    res_sparsities = []
    res_SRs = []
    activity_PRs = []

    rossler_timestep = 0.05

    @progress for (id,reservoir) in enumerate(reservoirs)


        input_size = 3
        output_size = 3 
        reservoir_size = size(reservoir)[1]

        traindata = train_data[:,1:end-1]
        targetdata = train_data[:,2:end]

        measurements = []

        # trying different setups of input and output layers
        for ii in 1:num_simulation_steps

            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size);
            reservoir_states = zeros(reservoir_size, size(traindata)[2])

            for node_id in node_ids[id]
                input_weights[node_id,:] .= 0
                reservoir[node_id,:] .= 0
                reservoir[:,node_id] .= 0
            end

            res_eigenvalues = eigvals(Matrix(reservoir))
            push!(res_PRs, Main.Reservoirs_src.participation_ratio(abs.(res_eigenvalues)))

            res_sparsity = Main.ConnectomeFunctions.compute_sparsity(reservoir)
            push!(res_sparsities, res_sparsity)

            res_SR = maximum(abs.(res_eigenvalues))
            push!(res_SRs, res_SR)

            # Initialising the reservoir
            current_state = zeros(reservoir_size)   


            # Training the output layer
            for i in 1:size(traindata, 2)-1
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, traindata[:, i], leak_rate)
                reservoir_states[:, i] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, targetdata, regularization_coefficient)
            

            test_id = rand(1:6)
            testdata = test_data[test_id]

            # Initialising predictions
            predictions = zeros(output_size, size(testdata,2))

            # Starting with the last state from training
            last_state = reservoir_states[:, end]

            reservoir_internal_states = zeros(reservoir_size, size(testdata,2))

            for i in 1:size(testdata,2)
                # Generate the next state
                if i < 100
                    input = testdata[:,i]
                else
                    input = predictions[:, i-1]
                end
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
                predicted_output = output_weights' * next_state
                reservoir_internal_states[:,i] = next_state
                # Storing the predicted output
                predictions[:, i] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            _, ev, _ = Main.Reservoirs_src.pca(reservoir_internal_states)

            push!(activity_PRs, Main.Reservoirs_src.participation_ratio(abs.(ev)))

            
            laststate = reservoir_internal_states[:,end]

            roscriticality = max_lyapunov_exponent(laststate, size(testdata)[2], reservoir, input_weights, testdata; ε=1e-7)

            push!(total_criticalities, roscriticality)

            # measure of performance
            t1, measure = Main.Reservoirs_src.valid_time_rossler(threshhold, predictions, testdata[:,1:end], rossler_timestep)

            push!(total_performances, measure)

        end


    end

    return total_performances, res_PRs, res_sparsities, res_SRs, activity_PRs, total_criticalities

end











function reservoir_dynamics(x, W, W_in, u)
    """
    Reservoir dynamics with input.

    Parameters:
    - x: Current reservoir state
    - W: Recurrent weight matrix
    - W_in: Input weight matrix
    - u: Current input vector

    Returns:
    - Updated reservoir state
    """
    return tanh.(W * x + W_in * u)
end



function max_lyapunov_exponent(state, T, W, W_in, input_signal; ε=1e-7)
    """
    Compute the largest Lyapunov exponent with input-driven dynamics.

    Parameters:
    - state: Initial state vector (N-dimensional)
    - dynamics: Function defining the reservoir dynamics f(state, W, W_in, input)
    - T: Total number of timesteps
    - W: Recurrent weight matrix
    - W_in: Input weight matrix
    - input_signal: Input matrix (M x T), where M is the input dimension
    - ε: Small perturbation magnitude

    Returns:
    - λ_max: Largest Lyapunov exponent
    """
    N = length(state)                   # Dimension of the state space
    total_steps = size(input_signal,2) # Input dimensions
    @assert T <= total_steps            # Ensure input signal is long enough
    
    perturbation = ε * randn(N)         # Initial perturbation vector
    divergence_sum = 0.0                # Accumulated divergence
    
    for t in 1:T
        u_t = input_signal[:, t]        # Get input vector at time t
        
        # Evolve the system and the perturbed state
        state_next = reservoir_dynamics(state, W, W_in, u_t)
        perturbed_next = reservoir_dynamics(state + perturbation, W, W_in, u_t)
        
        # Compute the new perturbation vector
        new_perturbation = perturbed_next - state_next
        
        # QR decomposition to orthogonalize
        _, R = qr(new_perturbation)
        norm_factor = abs(R[1, 1])      # Extract stretching factor
        divergence_sum += log(norm_factor)
        
        # Normalize the perturbation
        perturbation = new_perturbation / norm_factor
        
        # Update the state
        state = state_next
    end
    
    # Compute the largest Lyapunov exponent
    λ_max = divergence_sum / T
    return λ_max
end







function res_performance_memory_wout(reservoirs, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate)

    #parameters
    reservoir_size = size(reservoirs[1])[1]
    input_size = 1
    output_size = 100  # N output neurons
    train_length = 4000
    test_length = 1000

    energies = []

    memories = []
    total_performances = []
    total_errors = []
    Wouts = []

    @progress for reservoir in reservoirs
        mems = []

        for i in 1:num_simulation_steps
            mem = []

            # Create input and output weights
            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            # Generate random input sequence
            X_train = rand(train_length) .- 0.5
            X_test = rand(test_length) .- 0.5

            # Generate target data for training the output layer
            target_data = Main.Reservoirs_src.generate_memory_target_data(X_train, output_size)'


            # Update reservoir state for training
            reservoir_state_train = zeros(reservoir_size, train_length)

            current_state = zeros(reservoir_size)   

            for t in 1:train_length
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, X_train[t], leak_rate)
                reservoir_state_train[:, t] = current_state
            end

            # Train output weights
            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_train, target_data, regularization_coefficient)

            push!(Wouts, output_weights)
            # push!(energies, sum(reservoir_state_train.^2))
            push!(energies, mean(reservoir_state_train.^2))
            # println("Activity length: ", size(reservoir_state_train, 2))

        end

    end

    return Wouts, energies

                    
end


function res_performance_recall_wout(reservoirs, num_simulation_steps, input_scaling, reg, leak, Ls; threshold=0.8)
    recalls = []  # store best L per reservoir
    corr_vals = []

    energies = []

    Wouts = []

    @progress for res in reservoirs
        rr = []
        pc1 = []
        
        for kk in 1:num_simulation_steps
            best_L = NaN
            best_wout = nothing
            b_energy = nothing
            pc = []
            for L in Ls
                energy, wout, pf, _ = res_performance_recall_L_wout([res], num_simulation_steps, input_scaling, reg, leak, L)
                perf = mean(pf)
                push!(pc, perf)

                if perf < threshold
                    break  # stop increasing L once performance drops
                end
                best_L = L
                best_wout = wout
                b_energy = energy
            end
            push!(Wouts, best_wout)
            push!(energies, b_energy)

        end
    end

    return Wouts, energies
end

function res_performance_recall_L_wout(reservoirs, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate, L)   

    energies = []
    #parameters
    res_size = size(reservoirs[1])[1]

    input_dim = 2
    output_dim = 1
    n_trials_train = 100  # Number of training trials
    n_trials_test = 20  # Number of testing trials
    
    recall_scores = []
    total_performances = []
    total_errors = []

    Wouts = []


    @progress for reservoir in reservoirs
        recalls1 = []

        for i in 1:num_simulation_steps

            # Create weights
            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)

            

            x1_train_data = []
            x2_train_data = []
            x3_train_targets = []

            for x in 1:n_trials_train
                x1,x2,x3 = create_data(L)
                push!(x1_train_data, x1)
                push!(x2_train_data, x2)
                push!(x3_train_targets,x3)
            end

            x1_train_data = vcat(x1_train_data...)
            x2_train_data = vcat(x2_train_data...)
            x3_train_targets = vcat(x3_train_targets...)


            input_data = hcat(x1_train_data,x2_train_data)
            input_data = input_data'

            input_weights*input_data[:,100]

            reservoir_state_trains = zeros(res_size,length(x1_train_data))
            current_state = zeros(res_size) 

            for t in 1:length(x1_train_data)
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input_data[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, x3_train_targets', regularization_coefficient)
            # training_output = output_weights' * reservoir_state_trains

            push!(Wouts, output_weights)
            # push!(energies, sum(reservoir_state_trains.^2))
            push!(energies, mean(reservoir_state_trains.^2))
            # println("Activity length: ", size(reservoir_state_trains, 2))

            
            x1_test_data = []
            x2_test_data = []
            x3_test_targets = []

            for x in 1:n_trials_test
                x1,x2,x3 = create_data(L)
                push!(x1_test_data, x1)
                push!(x2_test_data, x2)
                push!(x3_test_targets,x3)
            end

            x1_test_data = vcat(x1_test_data...)
            x2_test_data = vcat(x2_test_data...)
            x3_test_targets = vcat(x3_test_targets...)


            input_data_test = hcat(x1_test_data,x2_test_data)
            input_data_test = input_data_test'


            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, length(x3_test_targets))


            for t in 1:length(x1_test_data)
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input_data_test[:,t], leak_rate)

                predicted_output = output_weights' * next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end


            recall_score = cor(x3_test_targets,final_outputs').^2
            recall_score = recall_score[1,1]

            push!(recalls1,recall_score)
            push!(total_performances, recall_score)

            nrmse_value = compute_rec_nrmse(x3_test_targets, final_outputs',L)
            push!(total_errors,nrmse_value)
        end
        push!(recall_scores, mean(recalls1))
    end
    return energies, Wouts, total_performances, total_errors
  
end


function res_performance_decisionmaking_wout(reservoirs, num_simulation_steps, input_scaling, reg, leak, biases; threshold=0.8)
    dms = []  # store best score per reservoir

    energies = []
    Wouts = []

    x_biases = 1.0 .- biases

    @progress for res in reservoirs
        rr = []
        for kk in 1:num_simulation_steps
            best_bias = NaN
            best_wout = nothing
            b_energy = nothing
            for (id,bias) in enumerate(biases)
                energy, wout, pf, _ = res_performance_decisionmaking_L_wout([res], num_simulation_steps, bias, input_scaling, reg, leak)
                perf = mean(pf)

                if perf < threshold
                    break  # stop increasing once performance drops
                end
                best_bias = x_biases[id]
                best_wout = wout
                b_energy = energy
            end
            push!(energies, b_energy)
            push!(Wouts, best_wout)
        end
    end

    return Wouts, energies
end

function res_performance_decisionmaking_L_wout(reservoirs, num_simulation_steps, bias ,input_scaling, regularization_coefficient, leak_rate)
    num_samples_train = 2000   # Number of training samples
    num_samples_test = 1000   # Number of testing samples
    switch_interval = 100      # Check for a possible switch every 100 timesteps
    input_dim = 2
    output_dim = 1
    res_size = size(reservoirs[1])[2]
    dm_scores = []
    total_performances = []
    total_errors = []
    Wouts = []
    energies = []

    @progress for reservoir in reservoirs

        dec_scores = []

        for i in 1:num_simulation_steps

            # Generate training data
            train_data, train_targets = Main.Reservoirs_src.generate_decisionmaking_data(num_samples_train, switch_interval, bias)
            test_data, test_targets = Main.Reservoirs_src.generate_decisionmaking_data(num_samples_test, switch_interval, bias)

            # Reservoir and input/output layers
            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)


            reservoir_state_trains = zeros(res_size,size(train_data)[2])
            current_state = zeros(res_size) 

            for t in 1:size(train_data)[2]
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, train_data[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, train_targets', regularization_coefficient)

            push!(Wouts, output_weights)
            # push!(energies, sum(reservoir_state_trains.^2))
            push!(energies, mean(reservoir_state_trains.^2))
            
            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, size(test_data)[2])


            for t in 1:size(test_data)[2]
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_data[:,t], leak_rate)

                # println("Next state: ", size(next_state))
                # println("Output weights: ", size(output_weights))
                predicted_output = output_weights' * next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end

            # dm_score = cor(test_targets,final_outputs').^2
            # dm_score = dm_score[1,1]

            dm_score = compute_dm_accuracy(final_outputs, test_targets', threshold=0.0)
            # dm_score = compute_dm_accuracy_window(final_outputs, test_targets', decision_window=switch_interval, threshold=0.0)

            push!(dec_scores, dm_score)
            push!(total_performances, dm_score)

            error_dm = compute_dm_nrmse(test_targets, final_outputs')

            # error_rate = compute_dm_error_rate(test_targets, final_outputs')
            push!(total_errors, error_dm)

        end

        push!(dm_scores,dec_scores)
    end

    return energies, Wouts, total_performances, total_errors

end

function res_performance_delay_decisionmaking_wout(reservoirs, num_simulation_steps, input_scaling, reg, leak, variances; threshold=0.8)
    ddms = []  # store best L per reservoir

    Wouts = []
    energies = []

    @progress for res in reservoirs
        rr = []
        for kk in 1:num_simulation_steps
            best_var = NaN
            best_wout = nothing
            b_energy = nothing
            for var in variances
                energy, wout, pf, _ = res_performance_delay_decisionmaking_L_wout([res], num_simulation_steps, var, input_scaling, reg, leak)
                perf = mean(pf)

                if perf < threshold
                    break  # stop increasing L once performance drops
                end
                best_var = var
                best_wout = wout
                b_energy = energy
            end
            push!(Wouts, best_wout)
            push!(energies, b_energy)
        end

    end

    return Wouts, energies
end

function res_performance_delay_decisionmaking_L_wout(reservoirs, num_simulation_steps, var, input_scaling, regularization_coefficient, leak_rate)

    # Parameters
    num_trials = 100         # Number of trials
    stim_duration = 20       # Duration of stimulus presentation (time steps)
    resp_duration = 10       # Duration of response period (time steps)


    ddm_scores = []
    total_performances = []
    total_errors = []

    input_dim = 2
    output_dim = 1

    Wouts = []
    energies = []

    res_size = size(reservoirs[1],2)

    for reservoir in reservoirs 

        dec_scores = []

        for i in 1:num_simulation_steps
            variance = var
            # Generate data
            training_inputs, training_target_output = Main.Reservoirs_src.generate_delay_decisionmaking_task_data(num_trials, stim_duration, resp_duration,variance)
            # Generate test data
            test_inputs, test_target_output = Main.Reservoirs_src.generate_delay_decisionmaking_task_data(10, stim_duration, resp_duration,variance)
            #

            input_weights = Main.Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(res_size, output_dim)

            push!(Wouts, output_weights)

            reservoir_state_trains = zeros(res_size,size(training_inputs)[2])
            current_state = zeros(res_size) 

            for t in 1:size(training_inputs)[2]
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, training_inputs[:,t], leak_rate)
                reservoir_state_trains[:, t] = current_state
            end
            # push!(energies, sum(reservoir_state_trains.^2))
            push!(energies, mean(reservoir_state_trains.^2))

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_state_trains, training_target_output', regularization_coefficient)
            training_output = output_weights' * reservoir_state_trains


            last_state = reservoir_state_trains[:, end]
            final_outputs = zeros(output_dim, size(test_inputs)[2])


            for t in 1:size(test_inputs)[2]
                next_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_inputs[:,t], leak_rate)

                predicted_output = output_weights' * next_state

                # Storing the predicted output
                final_outputs[t] = predicted_output
                
                # Updating the last state for the next prediction
                last_state = next_state
            end


            # ddm_score = cor(test_target_output,final_outputs').^2
            # ddm_score = ddm_score[1,1]

            trial_len = stim_duration + resp_duration  # or whatever full trial length is
            fix_len = stim_duration  # fixation/stimulus period to ignore
            ddm_score = delay_decision_accuracy_ignore_fixation(final_outputs[:], test_target_output, trial_len, fix_len)


            push!(dec_scores, ddm_score)
            push!(total_performances, ddm_score)
            # error_rate_filtered = compute_error_rate_ignore_fixation(test_target_output, final_outputs')

            error_rate_filtered = ddm_nrmse_ignore_fixation(vec(test_target_output), vec(final_outputs'), trial_len, fix_len)


            push!(total_errors, error_rate_filtered)
        end

        push!(ddm_scores, mean(dec_scores))

    end

    return energies, Wouts, total_performances, total_errors

end

function res_performance_oscillator_wout(reservoirs, fulldata, threshold, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate)

    total_performances = []
    total_errors = []
    savetimestep = 0.1
    train_period_length = 1000
    test_period_length = 2000

    Wouts = []
    energies = []

    train_data = fulldata[1:train_period_length]


    @progress for reservoir in reservoirs

        input_size = 1
        output_size = 1 
        reservoir_size = size(reservoir)[1]

        # trying different setups of input and output layers
        for i in 1:num_simulation_steps
            max_start = length(fulldata) - train_period_length - test_period_length + 1
            rr = rand(1:max_start)
            test_start = train_period_length + rr
            test_end = test_start + test_period_length - 1
            test_data = fulldata[test_start:test_end]

            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            # Initialize reservoir states
            reservoir_states = zeros(reservoir_size, length(train_data)-1)
            current_state = zeros(reservoir_size)
            
            # Training
            for i in 1:length(train_data)-1
                input = train_data[i]  # Input must be a vector
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
                reservoir_states[:, i] = current_state
            end
            
            # Train output weights
            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, train_data[2:end]', regularization_coefficient)
            push!(Wouts, output_weights)
            # push!(energies, sum(reservoir_states.^2))
            push!(energies, mean(reservoir_states.^2))

        end

    end

    return Wouts, energies

end


function res_performance_lotka_2d_wout(reservoirs, fulldata, threshold, num_simulation_steps, input_scaling, regularization_coefficient, leak_rate)
    total_performances = []
    total_errors = []
    dt = 0.1
    train_period_length = 1000
    test_period_length = 2000

    Wouts= []
    energies = []

    train_data = fulldata[1:train_period_length,:]'

    @progress for reservoir in reservoirs
        input_size = 2
        output_size = 2
        reservoir_size = size(reservoir, 1)

        for sim in 1:num_simulation_steps
            max_start = size(fulldata, 1) - train_period_length - test_period_length + 1
            rr = rand(1:max_start)
            test_start = train_period_length + rr
            test_end = test_start + test_period_length - 1
            test_data = fulldata[test_start:test_end,:]'

            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            # Initialize reservoir states
            reservoir_states = zeros(reservoir_size, size(train_data, 2)-1)
            current_state = zeros(reservoir_size)

            # Training
            for i in 1:size(train_data, 2)-1
                input = train_data[:, i]
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
                reservoir_states[:, i] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, train_data[:, 2:end], regularization_coefficient)
            push!(Wouts, output_weights)
            # push!(energies, sum(reservoir_states.^2))
            push!(energies, mean(reservoir_states.^2))
        end
    end

    return Wouts, energies
end

function res_performance_lorenz_wout(reservoirs, train_data, test_data, num_simulation_steps, threshhold, input_scaling, regularization_coefficient, leak_rate)
    res_performances = []
    total_performances = []
    total_errors = []
    lorenz_savetimestep = 0.02
    train_period_length= 2000
    test_period_length = 1000

    Wouts= []
    energies = []

    ross_timestep = 0.05

    @progress for reservoir in reservoirs
        #parameters
        input_size = 3
        output_size = 3 
        reservoir_size = size(reservoir,2)

        traindata = train_data[:,1:end-1]
        targetdata = train_data[:,2:end]


        # trying different setups of input and output layers
        for i in 1:num_simulation_steps
            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

            reservoir_states = zeros(reservoir_size, size(traindata,2))

            # Initialising the reservoir
            current_state = zeros(reservoir_size)   


            # Training the output layer
            for i in 1:size(traindata, 2)
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, traindata[:, i], leak_rate)
                reservoir_states[:, i] = current_state
            end

            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, targetdata, regularization_coefficient)
            push!(Wouts, output_weights)
            # push!(energies, sum(reservoir_states.^2))
            push!(energies, mean(reservoir_states.^2))

        end

    end

    return Wouts, energies
                            
end

function res_performance_rossler_wout(reservoirs, train_data, test_data, num_simulation_steps, threshhold, input_scaling, regularization_coefficient, leak_rate)

    res_performances = []
    total_performances = []
    total_errors = []
    rossler_savetimestep = 0.05
    train_period_length= 10000
    test_period_length = 5000

    Wouts = []
    energies = []

    @progress for reservoir in reservoirs

        input_size = 3
        output_size = 3 
        reservoir_size = size(reservoir)[1]

        traindata = train_data[:,1:end-1]
        targetdata = train_data[:,2:end]

        measurements = []

        # trying different setups of input and output layers
        for i in 1:num_simulation_steps

            input_weights = Main.Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
            output_weights = Main.Reservoirs_src.initialize_output_weights(reservoir_size, output_size);
            reservoir_states = zeros(reservoir_size, size(traindata)[2])

            # Initialising the reservoir
            current_state = zeros(reservoir_size)   


            # Training the output layer
            for i in 1:size(traindata, 2)-1
                current_state = Main.Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, traindata[:, i], leak_rate)
                reservoir_states[:, i] = current_state
            end

            
            output_weights = Main.Reservoirs_src.train_output_weights(reservoir_states, targetdata, regularization_coefficient)
            
            push!(Wouts, output_weights)
            # push!(energies, sum(reservoir_states.^2))
            push!(energies, mean(reservoir_states.^2))

        end

    end

    return Wouts, energies

end




end