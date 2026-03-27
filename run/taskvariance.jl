using Plots
using Plots.PlotMeasures
using TSne, Clustering, Distances
using Tables, Statistics
using ProgressLogging
using JLD2
using .Reservoirs_src

subnetwork_sizes = [size(mat[1],1) for mat in conn_ESNs]

# Task Variance Function
function task_variance(reservoir_output)

    # reservoir_output: matrix of size (num_units, num_trials, num_timepoints)
    num_units, _, _ = size(reservoir_output)
    
    # Initialize a vector to store task variance for each unit
    tv = zeros(num_units)
    
    # Loop over each unit
    for i in 1:num_units
        # Compute the mean activity of unit i across trials at each time point
        mean_activity_timepoint = mean(reservoir_output[i, :, :], dims=2)
        
        # Compute variance across trials for each time point
        variance_timepoint = mean((reservoir_output[i, :, :] .- mean_activity_timepoint).^2, dims=2)
        
        # Average variance across all time points
        tv[i] = mean(variance_timepoint)
    end
    
    return tv
end

# Fractional Task Variance Function
function fractional_task_variance(tv_A::Vector, tv_B::Vector)
    # tv_A and tv_B are task variances for two different tasks
    return (tv_A .- tv_B) ./ (tv_A .+ tv_B)
end


#############
# Weighted TV
#############

function weighted_memory_task_variance(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates)
    reservoir_size= size(reservoir)[2]

    train_length = 4000
    test_length = 1000
    input_size = 1
    output_size = 200


    reservoir_output_taskB = zeros(reservoir_size, test_length, num_trials)


    for trial in 1:num_trials
        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        # Create input weights
        input_weights = Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling)

        # Initialize output weights
        output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_size)

        # Generate random input sequence
        X_train = rand(train_length) .- 0.5
        X_test = rand(test_length) .- 0.5

        # Generate target data for training the output layer
        target_data = Reservoirs_src.generate_memory_target_data(X_train, output_size)'


        # Update reservoir state for training
        reservoir_state_train = zeros(reservoir_size, train_length)
        current_state = zeros(reservoir_size)   

        for t in 1:train_length
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, X_train[t], leak_rate)
            reservoir_state_train[:, t] = current_state
        end

        # Train output weights
        output_weights = Reservoirs_src.train_output_weights(reservoir_state_train, target_data, regularization_coefficient)


        # reservoir_state_test = zeros(reservoir_size, test_length)
        final_outputs = zeros(output_size, test_length)


        # Starting with the last state from training
        last_state = reservoir_state_train[:, end]

        for t in 1:test_length
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, X_test[t], leak_rate)

            predicted_output = output_weights' * next_state
            # Calculate the sum of outgoing weights for each node
            # outgoing_weights_sum = sum(output_weights, dims=2)
            outgoing_weights_sum = sum(abs.(output_weights), dims=2)
            
            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            
            # Store the modified reservoir state for this trial
            reservoir_output_taskB[:, t, trial] = weighted_state

            # Storing the predicted output
            final_outputs[:, t] = predicted_output
            
            # Updating the last state for the next prediction
            last_state = next_state
        end

    end

    for trial in 1:num_trials
        reservoir_output_taskB[:,:,trial] = reservoir_output_taskB[:,:,trial] ./ maximum(abs.(reservoir_output_taskB[:,:,trial]))
    end

    return task_variance(reservoir_output_taskB)
end

function weighted_recall_task_variance(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates)

    L = 40  # Length of the memorized sequence to recall
    input_dim = 2
    output_dim = 1

    n_trials_train = 200  # Number of training trials
    n_trials_test = 51  # Number of testing trials

    res_size= size(reservoir)[2]

    reservoir_output_taskB = zeros(res_size, (L*2+1)*n_trials_test, num_trials)

    function create_data(L)
        x1 = vcat([rand() for i in 1:L],[0 for i in 1:L+1])
        x2 = vcat([0 for i in 1:L],[1],[0 for i in 1:L])
        x3 = vcat([0 for i in 1:L+1],x1[1:L])
        return x1,x2,x3
    end

    for trial in 1:num_trials
        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        input_weights = Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
        output_weights = Reservoirs_src.initialize_output_weights(res_size, output_dim)


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

        reservoir_state_trains = zeros(res_size,length(x1_train_data))
        current_state = zeros(res_size) 

        for t in 1:length(x1_train_data)
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input_data[:,t], leak_rate)
            reservoir_state_trains[:, t] = current_state
        end

        output_weights = Reservoirs_src.train_output_weights(reservoir_state_trains, x3_train_targets', regularization_coefficient)
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
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input_data_test[:,t], leak_rate)

            predicted_output = output_weights' * next_state
            # Calculate the sum of outgoing weights for each node
            # outgoing_weights_sum = sum(output_weights, dims=2)
            outgoing_weights_sum = sum(abs.(output_weights), dims=2)
            
            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            
            # Store the modified reservoir state for this trial
            reservoir_output_taskB[:, t, trial] = weighted_state

            # Storing the predicted output
            final_outputs[t] = predicted_output
            
            # Updating the last state for the next prediction
            last_state = next_state
        end
    end

    for trial in 1:num_trials
        reservoir_output_taskB[:,:,trial] = reservoir_output_taskB[:,:,trial] ./ maximum(abs.(reservoir_output_taskB[:,:,trial]))
    end

    return task_variance(reservoir_output_taskB)
end

function weighted_recall_task_variance_L(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates, L)

    input_dim = 2
    output_dim = 1

    n_trials_train = 100  # Number of training trials
    n_trials_test = 31  # Number of testing trials

    res_size= size(reservoir)[2]

    reservoir_output_taskB = zeros(res_size, (L*2+1)*n_trials_test, num_trials)

    function create_data(L)
        x1 = vcat([rand() for i in 1:L],[0 for i in 1:L+1])
        x2 = vcat([0 for i in 1:L],[1],[0 for i in 1:L])
        x3 = vcat([0 for i in 1:L+1],x1[1:L])
        return x1,x2,x3
    end

    for trial in 1:num_trials
        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        input_weights = Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
        output_weights = Reservoirs_src.initialize_output_weights(res_size, output_dim)

        

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

        reservoir_state_trains = zeros(res_size,length(x1_train_data))
        current_state = zeros(res_size) 

        for t in 1:length(x1_train_data)
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input_data[:,t], leak_rate)
            reservoir_state_trains[:, t] = current_state
        end

        output_weights = Reservoirs_src.train_output_weights(reservoir_state_trains, x3_train_targets', regularization_coefficient)
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
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input_data_test[:,t], leak_rate)

            predicted_output = output_weights' * next_state
            # Calculate the sum of outgoing weights for each node
            # outgoing_weights_sum = sum(output_weights, dims=2)
            outgoing_weights_sum = sum(abs.(output_weights), dims=2)
            
            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            
            # Store the modified reservoir state for this trial
            reservoir_output_taskB[:, t, trial] = weighted_state

            # Storing the predicted output
            final_outputs[t] = predicted_output
            
            # Updating the last state for the next prediction
            last_state = next_state
        end
    end

    for trial in 1:num_trials
        reservoir_output_taskB[:,:,trial] = reservoir_output_taskB[:,:,trial] ./ maximum(abs.(reservoir_output_taskB[:,:,trial]))
    end

    return task_variance(reservoir_output_taskB)
end

function weighted_decisionmaking_task_variance(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates)
    reservoir_size= size(reservoir)[2]
    input_dim = 2
    output_dim = 1

    num_samples_train = 1000   # Number of training samples
    num_samples_test = 1000   # Number of testing samples
    switch_interval = 100
    reservoir_output_taskC = zeros(reservoir_size, num_samples_test, num_trials)

    for trial in 1:num_trials
        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        # Generate training data
        bias = rand()
        train_data, train_targets = Reservoirs_src.generate_decisionmaking_data(num_samples_train, switch_interval,bias)
        test_data, test_targets = Reservoirs_src.generate_decisionmaking_data(num_samples_test, switch_interval,bias)

        input_weights = Reservoirs_src.create_input_weights(input_dim, reservoir_size, input_scaling)
        output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_dim)

        reservoir_state_trains = zeros(reservoir_size,size(train_data)[2])
        current_state = zeros(reservoir_size) 

        for t in 1:size(train_data)[2]
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, train_data[:,t], leak_rate)
            reservoir_state_trains[:, t] = current_state
        end

        output_weights = Reservoirs_src.train_output_weights(reservoir_state_trains, train_targets', regularization_coefficient)
        training_output = output_weights' * reservoir_state_trains

        last_state = reservoir_state_trains[:, end]
        final_outputs = zeros(output_dim, size(test_data)[2])


        for t in 1:size(test_data)[2]
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_data[:,t], leak_rate)

            predicted_output = output_weights' * next_state
            # Calculate the sum of outgoing weights for each node
            # outgoing_weights_sum = sum(output_weights, dims=2)
            outgoing_weights_sum = sum(abs.(output_weights), dims=2)
            
            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            
            # Store the modified reservoir state for this trial
            reservoir_output_taskC[:, t, trial] = weighted_state

            # Storing the predicted output
            final_outputs[t] = predicted_output
            
            # Updating the last state for the next prediction
            last_state = next_state
        end

    end

    for trial in 1:num_trials
        reservoir_output_taskC[:,:,trial] = reservoir_output_taskC[:,:,trial] ./ maximum(abs.(reservoir_output_taskC[:,:,trial]))
    end

    return task_variance(reservoir_output_taskC)
end

function weighted_decisionmaking_task_variance_L(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates, L)
    reservoir_size= size(reservoir)[2]
    input_dim = 2
    output_dim = 1

    num_samples_train = 1000   # Number of training samples
    num_samples_test = 1000   # Number of testing samples
    switch_interval = 100
    reservoir_output_taskC = zeros(reservoir_size, num_samples_test, num_trials)

    for trial in 1:num_trials
        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        # Generate training data
        bias = L
        train_data, train_targets = Reservoirs_src.generate_decisionmaking_data(num_samples_train, switch_interval,bias)
        test_data, test_targets = Reservoirs_src.generate_decisionmaking_data(num_samples_test, switch_interval,bias)

        input_weights = Reservoirs_src.create_input_weights(input_dim, reservoir_size, input_scaling)
        output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_dim)


        reservoir_state_trains = zeros(reservoir_size,size(train_data)[2])
        current_state = zeros(reservoir_size) 

        for t in 1:size(train_data)[2]
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, train_data[:,t], leak_rate)
            reservoir_state_trains[:, t] = current_state
        end

        output_weights = Reservoirs_src.train_output_weights(reservoir_state_trains, train_targets', regularization_coefficient)
        training_output = output_weights' * reservoir_state_trains

        last_state = reservoir_state_trains[:, end]
        final_outputs = zeros(output_dim, size(test_data)[2])


        for t in 1:size(test_data)[2]
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_data[:,t], leak_rate)

            predicted_output = output_weights' * next_state
            # Calculate the sum of outgoing weights for each node
            # outgoing_weights_sum = sum(output_weights, dims=2)
            outgoing_weights_sum = sum(abs.(output_weights), dims=2)
            
            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            
            # Store the modified reservoir state for this trial
            reservoir_output_taskC[:, t, trial] = weighted_state

            # Storing the predicted output
            final_outputs[t] = predicted_output
            
            # Updating the last state for the next prediction
            last_state = next_state
        end

    end

    for trial in 1:num_trials
        reservoir_output_taskC[:,:,trial] = reservoir_output_taskC[:,:,trial] ./ maximum(abs.(reservoir_output_taskC[:,:,trial]))
    end

    return task_variance(reservoir_output_taskC)
end

function weighted_delay_decisionmaking_task_variance(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates)
    res_size = size(reservoir)[2]
    numtraintrials=50
    # Parameters
    stim_duration = 10       # Duration of stimulus presentation (time steps)
    resp_duration = 10       # Duration of response period (time steps)
    
    test_length = 10

    dim2 = (stim_duration + resp_duration) * test_length
    reservoir_output_taskC = zeros(res_size, dim2, num_trials)

    for trial in 1:num_trials

        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        variance = rand()*2         # Variance of the sensory evidence (somewhere between 0 and 2)

        # Generate data
        training_inputs, training_target_output = Reservoirs_src.generate_delay_decisionmaking_task_data(numtraintrials, stim_duration, resp_duration, variance)

        input_dim = 2
        output_dim = 1

        input_weights = Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
        output_weights = Reservoirs_src.initialize_output_weights(res_size, output_dim)

        reservoir_state_trains = zeros(res_size,size(training_inputs)[2])
        current_state = zeros(res_size) 

        for t in 1:size(training_inputs)[2]
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, training_inputs[:,t], leak_rate)
            reservoir_state_trains[:, t] = current_state
        end

        output_weights = Reservoirs_src.train_output_weights(reservoir_state_trains, training_target_output', regularization_coefficient)

        # Generate test data
        test_inputs, test_target_output = Reservoirs_src.generate_delay_decisionmaking_task_data(test_length, stim_duration, resp_duration,variance)
        #

        last_state = reservoir_state_trains[:, end]
        final_outputs = zeros(output_dim, size(test_inputs)[2])
        
        for t in 1:size(test_inputs)[2]
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_inputs[:,t], leak_rate)

            predicted_output = output_weights' * next_state

            # Storing the predicted output
            final_outputs[t] = predicted_output

            outgoing_weights_sum = sum(abs.(output_weights), dims=2)

            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            # Store the modified reservoir state for this trial
            reservoir_output_taskC[:, t, trial] = weighted_state
            
            # Updating the last state for the next prediction
            last_state = next_state
        end

    end

    for trial in 1:num_trials
        reservoir_output_taskC[:,:,trial] = reservoir_output_taskC[:,:,trial] ./ maximum(abs.(reservoir_output_taskC[:,:,trial]))
    end

    return task_variance(reservoir_output_taskC)

end

function weighted_delay_decisionmaking_task_variance_L(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates, L)
    res_size = size(reservoir)[2]
    numtraintrials=50
    # Parameters
    stim_duration = 10       # Duration of stimulus presentation (time steps)
    resp_duration = 10       # Duration of response period (time steps)
    
    test_length = 10

    dim2 = (stim_duration + resp_duration) * test_length
    reservoir_output_taskC = zeros(res_size, dim2, num_trials)

    for trial in 1:num_trials

        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        variance = L
        # Generate data
        training_inputs, training_target_output = Reservoirs_src.generate_delay_decisionmaking_task_data(numtraintrials, stim_duration, resp_duration, variance)

        input_dim = 2
        output_dim = 1

        input_weights = Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
        output_weights = Reservoirs_src.initialize_output_weights(res_size, output_dim)

        reservoir_state_trains = zeros(res_size,size(training_inputs)[2])
        current_state = zeros(res_size) 

        for t in 1:size(training_inputs)[2]
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, training_inputs[:,t], leak_rate)
            reservoir_state_trains[:, t] = current_state
        end


        output_weights = Reservoirs_src.train_output_weights(reservoir_state_trains, training_target_output', regularization_coefficient)

        # Generate test data
        test_inputs, test_target_output = Reservoirs_src.generate_delay_decisionmaking_task_data(test_length, stim_duration, resp_duration,variance)
        #

        last_state = reservoir_state_trains[:, end]
        final_outputs = zeros(output_dim, size(test_inputs)[2])
        
        for t in 1:size(test_inputs)[2]
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, test_inputs[:,t], leak_rate)

            predicted_output = output_weights' * next_state

            # Storing the predicted output
            final_outputs[t] = predicted_output

            outgoing_weights_sum = sum(abs.(output_weights), dims=2)

            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            # Store the modified reservoir state for this trial
            reservoir_output_taskC[:, t, trial] = weighted_state
            
            # Updating the last state for the next prediction
            last_state = next_state
        end

    end

    for trial in 1:num_trials
        reservoir_output_taskC[:,:,trial] = reservoir_output_taskC[:,:,trial] ./ maximum(abs.(reservoir_output_taskC[:,:,trial]))
    end

    return task_variance(reservoir_output_taskC)

end

function weighted_oscillator_task_variance(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates)

    # Define the function
    function periodic_function(x)
        return 3*sin(x) + 2*cos(2x) #+ 2*sin(1.5x)
    end

    # Generate time series data
    savetimestep = 0.1
    x_values = 0:savetimestep:10000  # Time steps
    fulldata = periodic_function.(x_values)

    train_period_length = 2000
    test_period_length = 1000

    train_data = fulldata[1:train_period_length]

    reservoir_size= size(reservoir)[2]
    input_size = 1
    output_size = 1

    train_period_length= 2000
    test_period_length = 1000


    reservoir_output_taskA = zeros(reservoir_size, test_period_length, num_trials)


    for trial in 1:num_trials
        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        input_weights = Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
        output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_size);
        reservoir_states = zeros(reservoir_size, length(train_data)-1)
        current_state = zeros(reservoir_size)

        # Training
        for i in 1:length(train_data)-1
            input = train_data[i]  # Input must be a vector
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
            reservoir_states[:, i] = current_state
        end

        output_weights = Reservoirs_src.train_output_weights(reservoir_states, train_data[2:end]', regularization_coefficient)

        # Initialize predictions
        predictions = zeros(output_size, test_period_length)
        last_state = reservoir_states[:, end]
        
        max_start = length(fulldata) - train_period_length - test_period_length + 1
        rr = rand(1:max_start)
        test_start = train_period_length + rr
        test_end = test_start + test_period_length - 1
        test_data = fulldata[test_start:test_end]

        # Predict using only previous predictions after 100 steps

        for t in 1:test_period_length
            # Generate the next state
            if t < 100
                input = test_data[t]
            else
                input = predictions[t-1]
            end
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
            
            # Calculate the sum of outgoing weights for each node
            # outgoing_weights_sum = sum(output_weights, dims=2)
            outgoing_weights_sum = sum(abs.(output_weights), dims=2)

            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            
            # Store the modified reservoir state for this trial
            reservoir_output_taskA[:, t, trial] = weighted_state
            
            predicted_output = output_weights' * next_state

            # Storing the predicted output
            predictions[t] = predicted_output[1]
            
            # Updating the last state for the next prediction
            last_state = next_state
        end
    end


    for trial in 1:num_trials
        reservoir_output_taskA[:,:,trial] = reservoir_output_taskA[:,:,trial] ./ maximum(abs.(reservoir_output_taskA[:,:,trial]))
    end

    return task_variance(reservoir_output_taskA)
end

function weighted_lotka_task_variance(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates)

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
    dt = 0.05

    prob = ODEProblem(lotka_volterra!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=dt)
    # fulldata = sol[1, :]  # Use only the prey population
    fulldata = reduce(hcat, sol.u)' # two populations

    # Split into train and test sets
    train_period_length = 2000
    test_period_length = 1000

    train_data = fulldata[1:train_period_length,:]'

    reservoir_size= size(reservoir)[2]
    input_size = 2
    output_size = 2

    reservoir_output_taskA = zeros(reservoir_size, test_period_length, num_trials)


    for trial in 1:num_trials
        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        input_weights = Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
        output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_size);
        reservoir_states = zeros(reservoir_size, size(train_data, 2)-1)
        
        current_state = zeros(reservoir_size)

        # Training
        for i in 1:size(train_data, 2)-1
            input = train_data[:,i]  # Input must be a vector
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
            reservoir_states[:, i] = current_state
        end

        output_weights = Reservoirs_src.train_output_weights(reservoir_states, train_data[:,2:end], regularization_coefficient)

        # Initialize predictions
        predictions = zeros(output_size, test_period_length)
        last_state = reservoir_states[:, end]

        max_start = size(fulldata,1) - train_period_length - test_period_length + 1
        rr = rand(1:max_start)
        test_start = train_period_length + rr
        test_end = test_start + test_period_length - 1
        test_data = fulldata[test_start:test_end,:]'

        # Predict using only previous predictions after 100 steps

        for t in 1:test_period_length
            # Generate the next state
            if t < 100
                input = test_data[:,t]
            else
                input = predictions[:,t-1]
            end
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
            
            # Calculate the sum of outgoing weights for each node
            # outgoing_weights_sum = sum(output_weights, dims=2)
            outgoing_weights_sum = sum(abs.(output_weights), dims=2)

            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            
            # Store the modified reservoir state for this trial
            reservoir_output_taskA[:, t, trial] = weighted_state
            
            predicted_output = output_weights' * next_state

            # Storing the predicted output
            predictions[:,t] = predicted_output
            
            # Updating the last state for the next prediction
            last_state = next_state
        end
    end


    for trial in 1:num_trials
        reservoir_output_taskA[:,:,trial] = reservoir_output_taskA[:,:,trial] ./ maximum(abs.(reservoir_output_taskA[:,:,trial]))
    end

    return task_variance(reservoir_output_taskA)
end

function weighted_lorenz_task_variance(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates)

    savetimestep=0.02
    prob = ODEProblem(Reservoirs_src.lorenz!, [1.0, 0.0, 0.0], (0.0, 200.0));
    lorenz_data = solve(prob, ABM54(), dt=0.02) #, saveat=savetimestep)
    lorenz_data = reduce(hcat, lorenz_data.u)
    lorenz_data = lorenz_data[:,300:end]

    train_period_length= 2000
    test_period_length = 1000
    lorenz_train_data, lorenz_test_data = Reservoirs_src.create_training_and_testing_periods(lorenz_data, train_period_length, test_period_length)

    lorenztraindata = lorenz_train_data[:,1:end-1]
    lorenztargetdata = lorenz_train_data[:,2:end]

    lorenztestdata=lorenz_test_data[2]


    reservoir_size= size(reservoir)[2]
    input_size = 3
    output_size = 3 

    train_period_length= 2000
    test_period_length = 1000


    reservoir_output_taskA = zeros(reservoir_size, test_period_length, num_trials)


    for trial in 1:num_trials
        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        input_weights = Reservoirs_src.create_input_weights(input_size, reservoir_size, input_scaling);
        output_weights = Reservoirs_src.initialize_output_weights(reservoir_size, output_size);
        reservoir_states = zeros(reservoir_size, size(lorenztraindata,2))

        # Initialising the reservoir
        current_state = zeros(reservoir_size)   


        # Training the output layer
        for i in 1:size(lorenztraindata, 2)-1
            current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, lorenztraindata[:, i], leak_rate)
            reservoir_states[:, i] = current_state
        end

        output_weights = Reservoirs_src.train_output_weights(reservoir_states, lorenztargetdata, regularization_coefficient)

        # Initialising predictions
        predictions = zeros(output_size, test_period_length)

        # Starting with the last state from training
        last_state = reservoir_states[:, end]

        idd = rand(1:5)
        lorenztestdata=lorenz_test_data[idd]


        for t in 1:test_period_length
            # Generate the next state
            if t < 100
                input = lorenztestdata[:,t]
            else
                input = predictions[:, t-1]
            end
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
            
            # Calculate the sum of outgoing weights for each node
            # outgoing_weights_sum = sum(output_weights, dims=2)
            outgoing_weights_sum = sum(abs.(output_weights), dims=2)

            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            
            # Store the modified reservoir state for this trial
            reservoir_output_taskA[:, t, trial] = weighted_state
            
            predicted_output = output_weights' * next_state

            # Storing the predicted output
            predictions[:, t] = predicted_output
            
            # Updating the last state for the next prediction
            last_state = next_state
        end
    end


    for trial in 1:num_trials
        reservoir_output_taskA[:,:,trial] = reservoir_output_taskA[:,:,trial] ./ maximum(abs.(reservoir_output_taskA[:,:,trial]))
    end

    return task_variance(reservoir_output_taskA)
end

function weighted_rossler_task_variance(reservoir, num_trials, input_scalings, regularization_coefficients, leak_rates)
    reservoir_size= size(reservoir)[2]
    # Parameters for the Rössler system
    a = 0.2
    b = 0.2
    c = 5.7
    p = (a, b, c)

    # Initial conditions and time span
    u0 = [1.0, 1.0, 1.0]
    tspan = (0.0, 300.0)

    # Solve the system
    prob = ODEProblem(roessler!, u0, tspan, p)
    rossler_sol = solve(prob, Tsit5(), saveat=0.02)
    rossler_data = reduce(hcat, rossler_sol.u)
    rossler_data = rossler_data[:,300:end]


    train_period_length= 3000
    test_period_length = 1000
    rossler_train_data, rossler_test_data = Reservoirs_src.create_training_and_testing_periods(rossler_data, train_period_length, test_period_length)
    
    input_size = 3
    output_size = 3
    
    reservoir_output_taskA = zeros(reservoir_size, test_period_length, num_trials)


    for trial in 1:num_trials
        input_scaling = input_scalings[trial]
        leak_rate = leak_rates[trial]
        regularization_coefficient = regularization_coefficients[trial]

        rosslertraindata = rossler_train_data[:,1:end-1]
        rosslertargetdata = rossler_train_data[:,2:end]

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

        # Initialising predictions
        predictions = zeros(output_size, test_period_length)

        # Starting with the last state from training
        last_state = reservoir_states[:, end]

        idd = rand(1:5)
        rosslertestdata=rossler_test_data[idd]


        for t in 1:test_period_length
            # Generate the next state
            if t < 100
                input = rosslertestdata[:,t]
            else
                input = predictions[:, t-1]
            end
            next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input, leak_rate)
            predicted_output = output_weights' * next_state

            # Calculate the sum of outgoing weights for each node
            # outgoing_weights_sum = sum(output_weights, dims=2)
            outgoing_weights_sum = sum(abs.(output_weights), dims=2)
            
            # Multiply each node's activity by its outgoing weight sum
            weighted_state = next_state .* outgoing_weights_sum
            
            # Store the modified reservoir state for this trial
            reservoir_output_taskA[:, t, trial] = weighted_state

            # Storing the predicted output
            predictions[:, t] = predicted_output
            
            # Updating the last state for the next prediction
            last_state = next_state
        end


    end

    for trial in 1:num_trials
        reservoir_output_taskA[:,:,trial] = reservoir_output_taskA[:,:,trial] ./ maximum(abs.(reservoir_output_taskA[:,:,trial]))
    end

    return task_variance(reservoir_output_taskA)
end

####

numres = 30

using .Reservoirs_src, .ConnectomeFunctions, ProgressLogging


# load the files from loadfiles.jl

# weighted_tv_conn = Vector{Any}(undef, 39)
# weighted_tv_rand = Vector{Any}(undef, 39)
# weighted_tv_cfg = Vector{Any}(undef, 39)

# ##########
# # Conn weighted TV
# ##########

# using Main.Threads

# @threads for i in 3:39
#     tvmem  = Vector{Any}(undef, numres)
#     tvrec  = Vector{Any}(undef, numres)
#     tvdm   = Vector{Any}(undef, numres)
#     tvddm  = Vector{Any}(undef, numres)
#     tvosc  = Vector{Any}(undef, numres)
#     tvlot  = Vector{Any}(undef, numres)
#     tvlor  = Vector{Any}(undef, numres)
#     tvros  = Vector{Any}(undef, numres)
    
#     subnetworks1 = ConnectomeFunctions.scale_matrices(deepcopy(conn_ESNs[i][1:numres]), 0.99)
#     subnetworks2 = ConnectomeFunctions.scale_matrices(deepcopy(conn_ESNs[i][1:numres]), 0.99)
#     subnetworks3 = ConnectomeFunctions.scale_matrices(deepcopy(conn_ESNs[i][1:numres]), 0.99)
#     subnetworks4 = ConnectomeFunctions.scale_matrices(deepcopy(conn_ESNs[i][1:numres]), 0.99)
#     subnetworks5 = ConnectomeFunctions.scale_matrices(deepcopy(conn_ESNs[i][1:numres]), 0.99)
#     subnetworks6 = ConnectomeFunctions.scale_matrices(deepcopy(conn_ESNs[i][1:numres]), 0.99)
#     subnetworks7 = ConnectomeFunctions.scale_matrices(deepcopy(conn_ESNs[i][1:numres]), 0.99)
#     subnetworks8 = ConnectomeFunctions.scale_matrices(deepcopy(conn_ESNs[i][1:numres]), 0.99)


#     for j in 1:numres

#         mem_is, rec_is, dm_is, ddm_is, osc_is, lot_is, lor_is, ross_is = input_conn[i,1], input_conn[i,2], input_conn[i,4], input_conn[i,4], input_conn[i,5], input_conn[i,6], input_conn[i,7], input_conn[i,8]
#         leak_rate_mem, leak_rate_rec, leak_rate_dm, leak_rate_ddm, leak_rate_osc, leak_rate_lot, leak_rate_lor, leak_rate_ross = leak_conn[i,1], leak_conn[i,2], leak_conn[i,3], leak_conn[i,4], leak_conn[i,5], leak_conn[i,6], leak_conn[i,7], leak_conn[i,8]
#         reg_coefficient_mem, reg_coefficient_rec, reg_coefficient_dm, reg_coefficient_ddm, reg_coefficient_osc, reg_coefficient_lot, reg_coefficient_lor, reg_coefficient_ross = reg_conn[i,1], reg_conn[i,2], reg_conn[i,3], reg_conn[i,4], reg_conn[i,5], reg_conn[i,6], reg_conn[i,7], reg_conn[i,8]

#         wtv_mem = weighted_memory_task_variance(subnetworks1[j],l, mem_is, reg_coefficient_mem, leak_rate_mem)
#         wtv_rec = weighted_recall_task_variance_L(subnetworks2[j],l, rec_is, reg_coefficient_rec, leak_rate_rec, conn_rec_Ls[i])
#         wtv_dm = weighted_decisionmaking_task_variance_L(subnetworks3[j],l, dm_is, reg_coefficient_dm, leak_rate_dm, conn_dm_Ls[i])
#         wtv_ddm = weighted_delay_decisionmaking_task_variance_L(subnetworks4[j],l, ddm_is, reg_coefficient_ddm, leak_rate_ddm, conn_ddm_Ls[i])
#         wtv_osc = weighted_oscillator_task_variance(subnetworks5[j],l, osc_is, reg_coefficient_osc, leak_rate_osc)
#         wtv_lot = weighted_lotka_task_variance(subnetworks6[j],l, lot_is, reg_coefficient_lot, leak_rate_lot)
#         wtv_lor = weighted_lorenz_task_variance(subnetworks7[j],l, lor_is, reg_coefficient_lor, leak_rate_lor)
#         wtv_ros = weighted_rossler_task_variance(subnetworks8[j],l, ross_is, reg_coefficient_ross, leak_rate_ross)

#         tvmem[j] = wtv_mem
#         tvrec[j] = wtv_rec
#         tvdm[j] = wtv_dm
#         tvddm[j] = wtv_ddm
#         tvosc[j] = wtv_osc
#         tvlot[j] = wtv_lot
#         tvlor[j] = wtv_lor
#         tvros[j] = wtv_ros

#     end

#     v = [tvmem,tvrec,tvdm,tvddm,tvosc,tvlot,tvlor,tvros]

#     weighted_tv_conn[i] = v
# end


# ##########
# # ER model weighted TV
# ##########

# #TV
# @threads for i in 3:39
#     tvmem  = Vector{Any}(undef, numres)
#     tvrec  = Vector{Any}(undef, numres)
#     tvdm   = Vector{Any}(undef, numres)
#     tvddm  = Vector{Any}(undef, numres)
#     tvosc  = Vector{Any}(undef, numres)
#     tvlot  = Vector{Any}(undef, numres)
#     tvlor  = Vector{Any}(undef, numres)
#     tvros  = Vector{Any}(undef, numres)
    
#     subnetworks1 = ConnectomeFunctions.scale_matrices(deepcopy(er_ESNs[i][1:numres]), 0.99)
#     subnetworks2 = ConnectomeFunctions.scale_matrices(deepcopy(er_ESNs[i][1:numres]), 0.99)
#     subnetworks3 = ConnectomeFunctions.scale_matrices(deepcopy(er_ESNs[i][1:numres]), 0.99)
#     subnetworks4 = ConnectomeFunctions.scale_matrices(deepcopy(er_ESNs[i][1:numres]), 0.99)
#     subnetworks5 = ConnectomeFunctions.scale_matrices(deepcopy(er_ESNs[i][1:numres]), 0.99)
#     subnetworks6 = ConnectomeFunctions.scale_matrices(deepcopy(er_ESNs[i][1:numres]), 0.99)
#     subnetworks7 = ConnectomeFunctions.scale_matrices(deepcopy(er_ESNs[i][1:numres]), 0.99)
#     subnetworks8 = ConnectomeFunctions.scale_matrices(deepcopy(er_ESNs[i][1:numres]), 0.99)

    

#     for j in 1:numres

#         mem_is, rec_is, dm_is, ddm_is, osc_is, lot_is, lor_is, ross_is = input_er[i,1], input_er[i,2], input_er[i,4], input_er[i,4], input_er[i,5], input_er[i,6], input_er[i,7], input_conn[i,8]
#         leak_rate_mem, leak_rate_rec, leak_rate_dm, leak_rate_ddm, leak_rate_osc, leak_rate_lot, leak_rate_lor, leak_rate_ross = leak_er[i,1], leak_er[i,2], leak_er[i,3], leak_er[i,4], leak_er[i,5], leak_er[i,6], leak_er[i,7], leak_conn[i,8]
#         reg_coefficient_mem, reg_coefficient_rec, reg_coefficient_dm, reg_coefficient_ddm, reg_coefficient_osc, reg_coefficient_lot, reg_coefficient_lor, reg_coefficient_ross = reg_er[i,1], reg_er[i,2], reg_er[i,3], reg_er[i,4], reg_er[i,5], reg_er[i,6], reg_er[i,7], reg_conn[i,8]

#         wtv_mem = weighted_memory_task_variance(subnetworks1[j],l, mem_is, reg_coefficient_mem, leak_rate_mem)
#         wtv_rec = weighted_recall_task_variance_L(subnetworks2[j],l, rec_is, reg_coefficient_rec, leak_rate_rec, er_rec_Ls[i])
#         wtv_dm = weighted_decisionmaking_task_variance_L(subnetworks3[j],l, dm_is, reg_coefficient_dm, leak_rate_dm, er_dm_Ls[i])
#         wtv_ddm = weighted_delay_decisionmaking_task_variance_L(subnetworks4[j],l, ddm_is, reg_coefficient_ddm, leak_rate_ddm, er_ddm_Ls[i])
#         wtv_osc = weighted_oscillator_task_variance(subnetworks5[j],l, osc_is, reg_coefficient_osc, leak_rate_osc)
#         wtv_lot = weighted_lotka_task_variance(subnetworks6[j],l, lot_is, reg_coefficient_lot, leak_rate_lot)
#         wtv_lor = weighted_lorenz_task_variance(subnetworks7[j],l, lor_is, reg_coefficient_lor, leak_rate_lor)
#         wtv_ros = weighted_rossler_task_variance(subnetworks8[j],l, ross_is, reg_coefficient_ross, leak_rate_ross)

#         tvmem[j] = wtv_mem
#         tvrec[j] = wtv_rec
#         tvdm[j] = wtv_dm
#         tvddm[j] = wtv_ddm
#         tvosc[j] = wtv_osc
#         tvlot[j] = wtv_lot
#         tvlor[j] = wtv_lor
#         tvros[j] = wtv_ros

#     end

#     v = [tvmem,tvrec,tvdm,tvddm,tvosc,tvlot,tvlor,tvros]

#     weighted_tv_rand[i] = v
# end

# ##########
# # CFG model weighted TV
# ##########

# #TV
# @threads for i in 3:39
#     tvmem  = Vector{Any}(undef, numres)
#     tvrec  = Vector{Any}(undef, numres)
#     tvdm   = Vector{Any}(undef, numres)
#     tvddm  = Vector{Any}(undef, numres)
#     tvosc  = Vector{Any}(undef, numres)
#     tvlot  = Vector{Any}(undef, numres)
#     tvlor  = Vector{Any}(undef, numres)
#     tvros  = Vector{Any}(undef, numres)
    
#     subnetworks1 = ConnectomeFunctions.scale_matrices(deepcopy(cfg_ESNs[i][1:numres]), 0.99)
#     subnetworks2 = ConnectomeFunctions.scale_matrices(deepcopy(cfg_ESNs[i][1:numres]), 0.99)
#     subnetworks3 = ConnectomeFunctions.scale_matrices(deepcopy(cfg_ESNs[i][1:numres]), 0.99)
#     subnetworks4 = ConnectomeFunctions.scale_matrices(deepcopy(cfg_ESNs[i][1:numres]), 0.99)
#     subnetworks5 = ConnectomeFunctions.scale_matrices(deepcopy(cfg_ESNs[i][1:numres]), 0.99)
#     subnetworks6 = ConnectomeFunctions.scale_matrices(deepcopy(cfg_ESNs[i][1:numres]), 0.99)
#     subnetworks7 = ConnectomeFunctions.scale_matrices(deepcopy(cfg_ESNs[i][1:numres]), 0.99)
#     subnetworks8 = ConnectomeFunctions.scale_matrices(deepcopy(cfg_ESNs[i][1:numres]), 0.99)

    

#     for j in 1:numres


#         mem_is, rec_is, dm_is, ddm_is, osc_is, lot_is, lor_is, ross_is = input_cfg[i,1], input_cfg[i,2], input_cfg[i,3], input_conn[i,4], input_cfg[i,5], input_cfg[i,6], input_cfg[i,7], input_conn[i,8]
#         leak_rate_mem, leak_rate_rec, leak_rate_dm, leak_rate_ddm, leak_rate_osc, leak_rate_lot, leak_rate_lor, leak_rate_ross = leak_cfg[i,1], leak_cfg[i,2], leak_conn[i,3], leak_cfg[i,4], leak_cfg[i,5], leak_cfg[i,6], leak_cfg[i,7], leak_conn[i,8]
#         reg_coefficient_mem, reg_coefficient_rec, reg_coefficient_dm, reg_coefficient_ddm, reg_coefficient_osc, reg_coefficient_lot, reg_coefficient_lor, reg_coefficient_ross = reg_cfg[i,1], reg_cfg[i,2], reg_cfg[i,3], reg_conn[i,4], reg_cfg[i,5], reg_cfg[i,6], reg_cfg[i,7], reg_conn[i,8]

#         wtv_mem = weighted_memory_task_variance(subnetworks1[j],l, mem_is, reg_coefficient_mem, leak_rate_mem)
#         wtv_rec = weighted_recall_task_variance_L(subnetworks2[j],l, rec_is, reg_coefficient_rec, leak_rate_rec, cfg_rec_Ls[i])
#         wtv_dm = weighted_decisionmaking_task_variance_L(subnetworks3[j],l, dm_is, reg_coefficient_dm, leak_rate_dm, cfg_dm_Ls[i])
#         wtv_ddm = weighted_delay_decisionmaking_task_variance_L(subnetworks4[j],l, ddm_is, reg_coefficient_ddm, leak_rate_ddm, cfg_ddm_Ls[i])
#         wtv_osc = weighted_oscillator_task_variance(subnetworks5[j],l, osc_is, reg_coefficient_osc, leak_rate_osc)
#         wtv_lot = weighted_lotka_task_variance(subnetworks6[j],l, lot_is, reg_coefficient_lot, leak_rate_lot)
#         wtv_lor = weighted_lorenz_task_variance(subnetworks7[j],l, lor_is, reg_coefficient_lor, leak_rate_lor)
#         wtv_ros = weighted_rossler_task_variance(subnetworks8[j],l, ross_is, reg_coefficient_ross, leak_rate_ross)

#         tvmem[j] = wtv_mem
#         tvrec[j] = wtv_rec
#         tvdm[j] = wtv_dm
#         tvddm[j] = wtv_ddm
#         tvosc[j] = wtv_osc
#         tvlot[j] = wtv_lot
#         tvlor[j] = wtv_lor
#         tvros[j] = wtv_ros

#     end

#     v = [tvmem,tvrec,tvdm,tvddm,tvosc,tvlot,tvlor,tvros]

#     weighted_tv_cfg[i] = v
# end



########################
# plotting example WTV for a given subnetwork

vid = 3

vv_conn = [(weighted_tv_conn[vid][1][1]) ./ maximum((weighted_tv_conn[vid][1][1])),
        (weighted_tv_conn[vid][2][1]) ./ maximum((weighted_tv_conn[vid][2][1])),
        (weighted_tv_conn[vid][3][1]) ./ maximum((weighted_tv_conn[vid][3][1])),
        (weighted_tv_conn[vid][4][1]) ./ maximum((weighted_tv_conn[vid][4][1])),
        (weighted_tv_conn[vid][5][1]) ./ maximum((weighted_tv_conn[vid][5][1])),
        (weighted_tv_conn[vid][6][1]) ./ maximum((weighted_tv_conn[vid][6][1])),
        (weighted_tv_conn[vid][7][1]) ./ maximum((weighted_tv_conn[vid][7][1])),
        (weighted_tv_conn[vid][8][1]) ./ maximum((weighted_tv_conn[vid][8][1]))]

vv_rand = [(weighted_tv_rand[vid][1][1]) ./ maximum((weighted_tv_rand[vid][1][1])),
        (weighted_tv_rand[vid][2][1]) ./ maximum((weighted_tv_rand[vid][2][1])),
        (weighted_tv_rand[vid][3][1]) ./ maximum((weighted_tv_rand[vid][3][1])),
        (weighted_tv_rand[vid][4][1]) ./ maximum((weighted_tv_rand[vid][4][1])),
        (weighted_tv_rand[vid][5][1]) ./ maximum((weighted_tv_rand[vid][5][1])),
        (weighted_tv_rand[vid][6][1]) ./ maximum((weighted_tv_rand[vid][6][1])),
        (weighted_tv_rand[vid][7][1]) ./ maximum((weighted_tv_rand[vid][7][1])),
        (weighted_tv_rand[vid][8][1]) ./ maximum((weighted_tv_rand[vid][8][1]))]

vv_cfg = [(weighted_tv_cfg[vid][1][1]) ./ maximum((weighted_tv_cfg[vid][1][1])),
        (weighted_tv_cfg[vid][2][1]) ./ maximum((weighted_tv_cfg[vid][2][1])),
        (weighted_tv_cfg[vid][3][1]) ./ maximum((weighted_tv_cfg[vid][3][1])),
        (weighted_tv_cfg[vid][4][1]) ./ maximum((weighted_tv_cfg[vid][4][1])),
        (weighted_tv_cfg[vid][5][1]) ./ maximum((weighted_tv_cfg[vid][5][1])),
        (weighted_tv_cfg[vid][6][1]) ./ maximum((weighted_tv_cfg[vid][6][1])),
        (weighted_tv_cfg[vid][7][1]) ./ maximum((weighted_tv_cfg[vid][7][1])),
        (weighted_tv_cfg[vid][8][1]) ./ maximum((weighted_tv_cfg[vid][8][1]))]



vv_conns = []
vv_rands = []
vv_cfgs = []

numres=30
for vid in 1:39
    vv_conn1 = []
        vv_rand1 = []
        vv_cfg1 = []

    for j in 1:numres

        

        vv_conn = [(weighted_tv_conn[vid][1][j]) ./ maximum((weighted_tv_conn[vid][1][j])),
                (weighted_tv_conn[vid][2][j]) ./ maximum((weighted_tv_conn[vid][2][j])),
                (weighted_tv_conn[vid][3][j]) ./ maximum((weighted_tv_conn[vid][3][j])),
                (weighted_tv_conn[vid][4][j]) ./ maximum((weighted_tv_conn[vid][4][j])),
                (weighted_tv_conn[vid][5][j]) ./ maximum((weighted_tv_conn[vid][5][j])),
                (weighted_tv_conn[vid][6][j]) ./ maximum((weighted_tv_conn[vid][6][j])),
                (weighted_tv_conn[vid][7][j]) ./ maximum((weighted_tv_conn[vid][7][j])),
                (weighted_tv_conn[vid][8][j]) ./ maximum((weighted_tv_conn[vid][8][j]))]

        vv_rand = [(weighted_tv_rand[vid][1][j]) ./ maximum((weighted_tv_rand[vid][1][j])),
                (weighted_tv_rand[vid][2][j]) ./ maximum((weighted_tv_rand[vid][2][j])),
                (weighted_tv_rand[vid][3][j]) ./ maximum((weighted_tv_rand[vid][3][j])),
                (weighted_tv_rand[vid][4][j]) ./ maximum((weighted_tv_rand[vid][4][j])),
                (weighted_tv_rand[vid][5][j]) ./ maximum((weighted_tv_rand[vid][5][j])),
                (weighted_tv_rand[vid][6][j]) ./ maximum((weighted_tv_rand[vid][6][j])),
                (weighted_tv_rand[vid][7][j]) ./ maximum((weighted_tv_rand[vid][7][j])),
                (weighted_tv_rand[vid][8][j]) ./ maximum((weighted_tv_rand[vid][8][j]))]

        vv_cfg = [(weighted_tv_cfg[vid][1][j]) ./ maximum((weighted_tv_cfg[vid][1][j])),
                (weighted_tv_cfg[vid][2][j]) ./ maximum((weighted_tv_cfg[vid][2][j])),
                (weighted_tv_cfg[vid][3][j]) ./ maximum((weighted_tv_cfg[vid][3][j])),
                (weighted_tv_cfg[vid][4][j]) ./ maximum((weighted_tv_cfg[vid][4][j])),
                (weighted_tv_cfg[vid][5][j]) ./ maximum((weighted_tv_cfg[vid][5][j])),
                (weighted_tv_cfg[vid][6][j]) ./ maximum((weighted_tv_cfg[vid][6][j])),
                (weighted_tv_cfg[vid][7][j]) ./ maximum((weighted_tv_cfg[vid][7][j])),
                (weighted_tv_cfg[vid][8][j]) ./ maximum((weighted_tv_cfg[vid][8][j]))]

                push!(vv_conn1, hcat(vv_conn...))
                push!(vv_rand1, hcat(vv_rand...))
                push!(vv_cfg1, hcat(vv_cfg...))
    end


    push!(vv_conns, vv_conn1)
    push!(vv_rands, vv_rand1)
    push!(vv_cfgs, vv_cfg1)
    
end


conn_wtv_prs = zeros(39, 8)
er_wtv_prs = zeros(39, 8)
cfg_wtv_prs = zeros(39, 8)

for i in 1:8

    for j in 3:11
        pr_conn = []
        pr_er = []
        pr_cfg = []

        conn_wtv_prs[j,i] = mean(participation_ratio2(weighted_tv_conn[j][i]))
        er_wtv_prs[j,i] = mean(participation_ratio2(weighted_tv_rand[j][i]))
        cfg_wtv_prs[j,i] = mean(participation_ratio2(weighted_tv_cfg[j][i]))
    end

end



h1 = heatmap(vv_conns[11][3],xticks=1:8, cbar=false,title="Conn", ylabel="Node ID", xlabel="Task ID",size=(600,600))
h2 = heatmap(vv_rands[11][3],cbar=false, title="ER",ylabel="Node ID",  xlabel="Task ID")
h3 = heatmap(vv_cfgs[11][3],cbar=false, title="CFG", ylabel="Node ID", xlabel="Task ID")


histogram(vv_conns[3][3][:,2],bins=0:0.1:1,alpha=0.4)
histogram!(vv_rands[3][3][:,2],bins=0:0.1:1,alpha=0.4)

h4 = plot(h1,h2,h3, layout=(1,3),size=(1000,500), leftmargin=5mm, bottommargin=5mm) #, plot_title="Normalised Weighted \n Task Variance", leftmargin=5mm)


########################

function participation_ratioo(v::AbstractVector)
    # v_norm = v ./ sum(v)               # normalise by total contribution
    # return 1.0 / sum(v_norm.^2)        # PR = 1 / Σ p_i^2
    return sum(v)^2 / sum(v .^ 2)
end

function mean_participation_ratio(X)
    pr_values = [sum(x)^2 / sum(x .^ 2) for x in X]
    return mean(pr_values)
end

function participation_ratio2(X)
    pr_values = [sum(x)^2 / sum(x .^ 2) for x in X]
    return pr_values
end



conn_wtv_prs = zeros(39, 8)
er_wtv_prs = zeros(39, 8)
cfg_wtv_prs = zeros(39, 8)

weighted_tv_conn[3][1]

numress=30

for i in 1:8

    for j in 1:39
        pr_conn = []
        pr_er = []
        pr_cfg = []

        conn_wtv_prs[j,i] = mean(participation_ratio2(weighted_tv_conn[j][i][1:numress]))
        er_wtv_prs[j,i] = mean(participation_ratio2(weighted_tv_rand[j][i][1:numress]))
        cfg_wtv_prs[j,i] = mean(participation_ratio2(weighted_tv_cfg[j][i][1:numress]))
    end

end

sizess = [size(conn_ESNs[i][1], 1) for i in 1:39]

using Plots.PlotMeasures
plot(
    heatmap(conn_wtv_prs ./ sizess, xticks=1:8, cbar=false, title="Conn", titlefontcolor=:blue, ylabel="Subnetwork ID", xlabel="Task ID"),
    heatmap(er_wtv_prs ./ sizess, cbar=false, title="ER", titlefontcolor=:crimson, yaxis=false, xlabel="Task ID"),
    heatmap(cfg_wtv_prs ./ sizess, title="CFG", titlefontcolor=:orange,  yaxis=false,xlabel="Task ID"),
    layout=Plots.grid(1,3, widths=[0.3,0.3,0.4]), size=(1000,300), xticks=1:8 , margin=5mm)


dodgerblues = [
    RGB(0.0, 0.2, 0.6),  # dark
    RGB(0.12, 0.56, 1.0), # middle (DodgerBlue)
    RGB(0.68, 0.85, 1.0)  # light
]

crimsons = [
    RGB(0.55, 0.0, 0.1),  # dark
    RGB(0.86, 0.08, 0.24), # middle (Crimson)
    RGB(1.0, 0.6, 0.65)   # light
]

oranges = [
    RGB(0.8, 0.4, 0.0),  # dark
    RGB(1.0, 0.55, 0.0), # middle (Orange)
    RGB(1.0, 0.85, 0.6)  # light
]




####################

using StatsPlots
#function to generate jittered x-coordinates
# width controls the amount of jitter
jitter(x_base, n_points; width=0.4) = x_base .+ (rand(n_points) * 2 * width .- width)

# Create the initial plot canvas
# By creating the plot first, all subsequent calls can use scatter!
p = plot()

boxplot!(p, [1], [conn_wtv_prs[3:end,1] ./ sizess[3:end]], boxwidth=0.5, color=:dodgerblue4, label=false, alpha=0.6)
boxplot!(p, [2], [er_wtv_prs[3:end,1] ./ sizess[3:end]], boxwidth=0.5, color=:crimson, label=false, alpha=0.6)
boxplot!(p, [3], [cfg_wtv_prs[3:end,1] ./ sizess[3:end]], boxwidth=0.5, color=:orange, label=false, alpha=0.6)
boxplot!(p, [5], [conn_wtv_prs[3:end,2] ./ sizess[3:end]], boxwidth=0.5, color=:dodgerblue4, label=false, alpha=0.6)
boxplot!(p, [6], [er_wtv_prs[3:end,2] ./ sizess[3:end]], boxwidth=0.5, color=:crimson, label=false, alpha=0.6)
boxplot!(p, [7], [cfg_wtv_prs[3:end,2] ./ sizess[3:end]], boxwidth=0.5, color=:orange, label=false, alpha=0.6)
boxplot!(p, [9], [conn_wtv_prs[3:end,3] ./ sizess[3:end]], boxwidth=0.5, color=:dodgerblue4, label=false, alpha=0.6)
boxplot!(p, [10], [er_wtv_prs[3:end,3] ./ sizess[3:end]], boxwidth=0.5, color=:crimson, label=false, alpha=0.6)
boxplot!(p, [11], [cfg_wtv_prs[3:end,3] ./ sizess[3:end]], boxwidth=0.5, color=:orange, label=false, alpha=0.6)
boxplot!(p, [13], [conn_wtv_prs[3:end,4] ./ sizess[3:end]], boxwidth=0.5, color=:dodgerblue4, label=false, alpha=0.6)
boxplot!(p, [14], [er_wtv_prs[3:end,4] ./ sizess[3:end]], boxwidth=0.5, color=:crimson, label=false, alpha=0.6)
boxplot!(p, [15], [cfg_wtv_prs[3:end,4] ./ sizess[3:end]], boxwidth=0.5, color=:orange, label=false, alpha=0.6)
boxplot!(p, [17], [conn_wtv_prs[3:end,5] ./ sizess[3:end]], boxwidth=0.5, color=:dodgerblue4, label=false, alpha=0.6)
boxplot!(p, [18], [er_wtv_prs[3:end,5] ./ sizess[3:end]], boxwidth=0.5, color=:crimson, label=false, alpha=0.6)
boxplot!(p, [19], [cfg_wtv_prs[3:end,5] ./ sizess[3:end]], boxwidth=0.5, color=:orange, label=false, alpha=0.6)
boxplot!(p, [21], [conn_wtv_prs[3:end,6] ./ sizess[3:end]], boxwidth=0.5, color=:dodgerblue4, label=false, alpha=0.6)
boxplot!(p, [22], [er_wtv_prs[3:end,6] ./ sizess[3:end]], boxwidth=0.5, color=:crimson, label=false, alpha=0.6)
boxplot!(p, [23], [cfg_wtv_prs[3:end,6] ./ sizess[3:end]], boxwidth=0.5, color=:orange, label=false, alpha=0.6)
boxplot!(p, [25], [conn_wtv_prs[3:end,7] ./ sizess[3:end]], boxwidth=0.5, color=:dodgerblue4, label=false, alpha=0.6)
boxplot!(p, [26], [er_wtv_prs[3:end,7] ./ sizess[3:end]], boxwidth=0.5, color=:crimson, label=false, alpha=0.6)
boxplot!(p, [27], [cfg_wtv_prs[3:end,7] ./ sizess[3:end]], boxwidth=0.5, color=:orange, label=false, alpha=0.6)
boxplot!(p, [29], [conn_wtv_prs[3:end,8] ./ sizess[3:end]], boxwidth=0.5, color=:dodgerblue4, label=false, alpha=0.6)
boxplot!(p, [30], [er_wtv_prs[3:end,8] ./ sizess[3:end]], boxwidth=0.5, color=:crimson, label=false, alpha=0.6)
boxplot!(p, [31], [cfg_wtv_prs[3:end,8] ./ sizess[3:end]], boxwidth=0.5, color=:orange, label=false, alpha=0.6)

# Define the base x-positions and the number of data columns to plot
x_bases = [1, 5, 9, 13, 17, 21, 25, 29] # Base x for 'conn' data for each task
num_tasks = 8
p = plot()
# Loop through each task/column of data
for i in 1:num_tasks
    col = i
    x_base = x_bases[i]

    n_end = length(12:size(conn_wtv_prs, 1)) # Safely get number of points
    scatter!(p, jitter(x_base, n_end), conn_wtv_prs[12:end, col] ./ sizess[12:end], c=dodgerblues[3], markersize=3, markerstrokewidth=0, label=false)
    scatter!(p, jitter(x_base + 1, n_end), er_wtv_prs[12:end, col] ./ sizess[12:end], c=crimsons[3], markersize=3, markerstrokewidth=0, label=false)
    scatter!(p, jitter(x_base + 2, n_end), cfg_wtv_prs[12:end, col] ./ sizess[12:end], c=oranges[3], markersize=3, markerstrokewidth=0, label=false)

    
    # Group 2 (rows 3:11)
    scatter!(p, jitter(x_base, 10), conn_wtv_prs[3:11, col] ./ sizess[3:11], c=:dodgerblue4, markersize=3, markerstrokewidth=0, label=false)
    scatter!(p, jitter(x_base + 1, 10), er_wtv_prs[3:11, col] ./ sizess[3:11], c=:crimson, markersize=3, markerstrokewidth=0, label=false)
    scatter!(p, jitter(x_base + 2, 10), cfg_wtv_prs[3:11, col] ./ sizess[3:11], c=:orange, markersize=3, markerstrokewidth=0, label=false)


end

# Add vertical lines and final plot settings
vline!(p, [4, 8, 12, 16, 20, 24, 28], c=:black, linestyle=:dash, label=false, grid=false)
plot!(p, xticks=([2, 6, 10, 14, 18, 22, 26, 30], ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8"]), 
ylabel="Mean Participation Ratio of WTV")


scatter!(p, [NaN], [NaN], c=dodgerblues[2], label="Drosophila Larva", markerstrokewidth=0)
scatter!(p, [NaN], [NaN], c=crimsons[2], label="ER (Drosophila Larva)", markerstrokewidth=0)
scatter!(p, [NaN], [NaN], c=oranges[2], label="CFG (Drosophila Larva)   ", markerstrokewidth=0)

scatter!(p, [NaN], [NaN], c=dodgerblues[3], label="Drosophila Adult", markerstrokewidth=0)
scatter!(p, [NaN], [NaN], c=crimsons[3], label="ER (Drosophila Adult)", markerstrokewidth=0)
scatter!(p, [NaN], [NaN], c=oranges[3], label="CFG (Drosophila Adult)   ", markerstrokewidth=0)

plot!(title="Mean Participation Ratio of Weighted Task Variance \n (normalised by network size)",legend_columns=3, yticks=0.2:0.2:0.8, ylim=(0,0.9), margin=4mm, legend=false)

using Plots.PlotMeasures



tv_plot = plot(p, legend_plot, layout=grid(2,1,heights=[0.8,0.2]), size=(500,400),dpi=600)

using HypothesisTests

pvalue(SignedRankTest(conn_wtv_prs[3:11,1] ./ sizess[3:11], er_wtv_prs[3:11,1] ./ sizess[3:11]))
pvalue(SignedRankTest(conn_wtv_prs[12:end,1] ./ sizess[12:end], er_wtv_prs[12:end,1] ./ sizess[12:end]))

pvalue(SignedRankTest(conn_wtv_prs[3:11,2] ./ sizess[3:11], er_wtv_prs[3:11,2] ./ sizess[3:11]))
pvalue(SignedRankTest(conn_wtv_prs[12:end,2] ./ sizess[12:end], er_wtv_prs[12:end,2] ./ sizess[12:end]))

pvalue(SignedRankTest(conn_wtv_prs[3:11,3] ./ sizess[3:11], er_wtv_prs[3:11,3] ./ sizess[3:11]))
pvalue(SignedRankTest(conn_wtv_prs[12:end,3] ./ sizess[12:end], er_wtv_prs[12:end,3] ./ sizess[12:end]))

pvalue(SignedRankTest(conn_wtv_prs[3:11,4] ./ sizess[3:11], er_wtv_prs[3:11,4] ./ sizess[3:11]))
pvalue(SignedRankTest(conn_wtv_prs[12:end,4] ./ sizess[12:end], er_wtv_prs[12:end,4] ./ sizess[12:end]))

pvalue(SignedRankTest(conn_wtv_prs[3:11,5] ./ sizess[3:11], er_wtv_prs[3:11,5] ./ sizess[3:11]))
pvalue(SignedRankTest(conn_wtv_prs[12:end,5] ./ sizess[12:end], er_wtv_prs[12:end,5] ./ sizess[12:end]))

pvalue(SignedRankTest(conn_wtv_prs[3:11,6] ./ sizess[3:11], er_wtv_prs[3:11,6] ./ sizess[3:11]))
pvalue(SignedRankTest(conn_wtv_prs[12:end,6] ./ sizess[12:end], er_wtv_prs[12:end,6] ./ sizess[12:end]))

pvalue(SignedRankTest(conn_wtv_prs[3:11,7] ./ sizess[3:11], er_wtv_prs[3:11,7] ./ sizess[3:11]))
pvalue(SignedRankTest(conn_wtv_prs[12:end,7] ./ sizess[12:end], er_wtv_prs[12:end,7] ./ sizess[12:end]))

pvalue(SignedRankTest(conn_wtv_prs[3:11,8] ./ sizess[3:11], er_wtv_prs[3:11,8] ./ sizess[3:11]))
pvalue(SignedRankTest(conn_wtv_prs[12:end,8] ./ sizess[12:end], er_wtv_prs[12:end,8] ./ sizess[12:end]))  

## comparing connectome larva vs adult
pvalue(MannWhitneyUTest(conn_wtv_prs[3:11,1] ./ sizess[3:11], conn_wtv_prs[12:end,1] ./ sizess[12:end]))
pvalue(MannWhitneyUTest(conn_wtv_prs[3:11,2] ./ sizess[3:11], conn_wtv_prs[12:end,2] ./ sizess[12:end]))
pvalue(MannWhitneyUTest(conn_wtv_prs[3:11,3] ./ sizess[3:11], conn_wtv_prs[12:end,3] ./ sizess[12:end]))
pvalue(MannWhitneyUTest(conn_wtv_prs[3:11,4] ./ sizess[3:11], conn_wtv_prs[12:end,4] ./ sizess[12:end]))
pvalue(MannWhitneyUTest(conn_wtv_prs[3:11,5] ./ sizess[3:11], conn_wtv_prs[12:end,5] ./ sizess[12:end]))
pvalue(MannWhitneyUTest(conn_wtv_prs[3:11,6] ./ sizess[3:11], conn_wtv_prs[12:end,6] ./ sizess[12:end]))




###########

# Do PR of rows vs columns of WTV matrices

numres=30
conn_prs_cols = [[mean([participation_ratioo(vv_conns[k][j][i,:]) for i in 1:size(vv_conns[k][j],1)]) for j in 1:numres] for k in 1:39]
rand_prs_cols = [[mean([participation_ratioo(vv_rands[k][j][i,:]) for i in 1:size(vv_rands[k][j],1)]) for j in 1:numres] for k in 1:39]
cfg_prs_cols = [[mean([participation_ratioo(vv_cfgs[k][j][i,:]) for i in 1:size(vv_cfgs[k][j],1)]) for j in 1:numres] for k in 1:39]

conn_prs_rows = [[mean([participation_ratioo(vv_conns[k][j][:,i]) for i in 1:8]) for j in 1:numres] for k in 1:39]
rand_prs_rows = [[mean([participation_ratioo(vv_rands[k][j][:,i]) for i in 1:8]) for j in 1:numres] for k in 1:39]
cfg_prs_rows = [[mean([participation_ratioo(vv_cfgs[k][j][:,i]) for i in 1:8]) for j in 1:numres] for k in 1:39]


pvalue(SignedRankTest(mean.(conn_prs_rows[3:11]) ./ sizess[3:11], mean.(rand_prs_rows[3:11]) ./ sizess[3:11]))
pvalue(SignedRankTest(mean.(conn_prs_cols[3:11]) ./ 8, mean.(rand_prs_cols[3:11]) ./ 8))

pvalue(SignedRankTest(mean.(conn_prs_rows[12:end]) ./ sizess[12:end], mean.(rand_prs_rows[12:end]) ./ sizess[12:end]))
pvalue(SignedRankTest(mean.(conn_prs_cols[12:end]) ./ 8, mean.(rand_prs_cols[12:end]) ./ 8))


function ellipse_plot!(x0, y0, a, b; color=:black, alpha=0.3, angle_deg=0, npoints=200)
    θ = range(0, 2π; length=npoints)
    φ = deg2rad(angle_deg)
    xs = a .* cos.(θ)
    ys = b .* sin.(θ)
    # rotation matrix
    x_rot = x0 .+ xs .* cos(φ) .- ys .* sin(φ)
    y_rot = y0 .+ xs .* sin(φ) .+ ys .* cos(φ)
    plot!(x_rot, y_rot, color=color, lw=10, alpha=0.4, label=false, fillalpha=alpha, fillcolor=color)
end


using LinearAlgebra, Statistics

function covariance_ellipse!(x, y; nsig=2.5, color=:black, alpha=0.3, npoints=200)

    μ = [mean(x); mean(y)]
    Σ = cov(hcat(x, y))             # 2×2 covariance

    eig = eigen(Symmetric(Σ))
    V = eig.vectors
    λ = eig.values

    # sort eigenvalues descending
    order = sortperm(λ, rev=true)
    λ = λ[order]
    V = V[:,order]

    θ = range(0, 2π; length=npoints)
    circle = [cos.(θ) sin.(θ)]'

    # scale by sqrt eigenvalues
    ellipse = μ .+ V * Diagonal(nsig .* sqrt.(λ)) * circle

    plot!(ellipse[1,:], ellipse[2,:],
          color=color, lw=4,
          fillalpha=alpha, fillcolor=color,
          label=false)
end

function minimum_volume_enclosing_ellipse(x, y; tol=1e-6, maxiter=10_000)

    X = hcat(x, y)'              # 2 × N
    d, N = size(X)

    Q = vcat(X, ones(1,N))       # 3 × N
    u = fill(1/N, N)

    for _ in 1:maxiter

        X_u = Q * Diagonal(u) * Q'
        M = diag(Q' * inv(X_u) * Q)

        j = argmax(M)
        maxM = M[j]

        if maxM ≤ d + 1 + tol
            break
        end

        step = (maxM - d - 1) / ((d + 1)*(maxM - 1))
        new_u = (1 - step) .* u
        new_u[j] += step

        u = new_u
    end

    # centre
    c = X * u

    # shape matrix
    A = inv((X * Diagonal(u) * X') - c*c') / d

    return c, A
end

function plot_mve!(x, y; color=:black, alpha=0.3, npoints=200)

    c, A = minimum_volume_enclosing_ellipse(x, y)

    θ = range(0, 2π; length=npoints)
    circle = [cos.(θ) sin.(θ)]'

    # transform unit circle via A^{-1/2}
    eig = eigen(Symmetric(A))
    V = eig.vectors
    Λ = eig.values

    transform = V * Diagonal(1 ./ sqrt.(Λ)) * V'

    ellipse = transform * circle .+ c

    plot!(ellipse[1,:], ellipse[2,:],
          color=color, lw=6,
          fillalpha=alpha, fillcolor=color,
          label=false)
end




x_er = mean.(rand_prs_rows) ./ sizess
y_er = mean.(rand_prs_cols) ./ 8
x_conn = mean.(conn_prs_rows) ./ sizess
y_conn = mean.(conn_prs_cols) ./ 8
x_cfg = mean.(cfg_prs_rows) ./ sizess
y_cfg = mean.(cfg_prs_cols) ./ 8


mean.(rand_prs_rows)
mean.(rand_prs_cols)

horiz_er = maximum(mean.(rand_prs_rows) ./ sizess) - minimum(mean.(rand_prs_rows) ./ sizess)
vert_er = maximum(mean.(rand_prs_cols) ./ 8) - minimum(mean.(rand_prs_cols) ./ 8)
maj_er = sqrt(horiz_er^2 + vert_er^2)
horiz_conn = maximum(mean.(conn_prs_rows) ./ sizess) - minimum(mean.(conn_prs_rows) ./ sizess)
vert_conn = maximum(mean.(conn_prs_cols) ./ 8) - minimum(mean.(conn_prs_cols) ./ 8)
maj_conn = sqrt(horiz_conn^2 + vert_conn^2)
horiz_cfg = maximum(mean.(cfg_prs_rows) ./ sizess) - minimum(mean.(cfg_prs_rows) ./ sizess)
vert_cfg = maximum(mean.(cfg_prs_cols) ./ 8) - minimum(mean.(cfg_prs_cols) ./ 8)
maj_cfg = sqrt(horiz_cfg^2 + vert_cfg^2)



p = plot()
covariance_ellipse!(x_er[3:end], y_er[3:end], color=:red, alpha=0.4)
covariance_ellipse!(x_conn[3:end], y_conn[3:end], color=:dodgerblue4, alpha=0.2)
plot_mve!(x_er[3:end], y_er[3:end], color=:red, alpha=0.4)
plot_mve!(x_conn[3:end], y_conn[3:end], color=:dodgerblue4, alpha=0.2)
plot!(title="Sparsity – Specialism")
# ellipse_plot!(mean(x_er[3:end])+0.015, mean(y_er[3:end])+0.004, maj_er/2+0.01, 0.027, color=:red, alpha=0.5,angle_deg=38)
# ellipse_plot!(mean(x_cfg[3:end])-0.005, mean(y_cfg[3:end])-0.005, maj_cfg/2, 0.025, color=:orange, alpha=0.3,angle_deg=45)
# ellipse_plot!(mean(x_conn[3:end])+0.005, mean(y_conn[3:end])-0.0, maj_conn/2+0.01, 0.025, color=:dodgerblue4, alpha=0.1,angle_deg=45)
scatter!(x_er[12:end], y_er[12:end], label="ER", c=crimsons[3], markersize=8, markerstrokewidth=0.05)
scatter!(x_er[3:11], y_er[3:11], label=false, c=:crimson, markersize=8, markerstrokewidth=0,
         aspectratio=:equal, size=(600,600))
# scatter!(x_cfg[12:end], y_cfg[12:end], label="CFG", c=oranges[3], markersize=8, markerstrokewidth=0.1,
#          aspectratio=:equal, size=(600,600))
# scatter!(x_cfg[3:11], y_cfg[3:11], label=false, c=:orange, markersize=8, markerstrokewidth=0,
#          aspectratio=:equal, size=(600,600))
scatter!(x_conn[12:end], y_conn[12:end], label="Conn", c=dodgerblues[3], markersize=8, markerstrokewidth=0.1,
         aspectratio=:equal, size=(600,600))
scatter!(x_conn[3:11], y_conn[3:11], label=false, c=:dodgerblue4, markersize=8, markerstrokewidth=0,
         aspectratio=:equal, size=(600,600))
plot!(ytickfontsize=14,xtickfontsize=16,legendfontsize=14, titlefontsize=18, grid=false, xlim=(0.15,0.55), ylim=(0.3,0.6))


using StatsPlots

s1 = plot()
boxplot!(s1, [1], [mean.(conn_prs_cols[3:end]) ./ 8], c=:dodgerblue4, lw=3, alpha=0.4)
boxplot!(s1, [2], [mean.(rand_prs_cols[3:end]) ./ 8], c=:crimson, lw=3, alpha=0.4)
boxplot!(s1, [3], [mean.(cfg_prs_cols[3:end]) ./ 8], c=:orange, lw=3, alpha=0.4)
scatter!(s1, jitter(1, 28), mean.(conn_prs_cols[12:end] ./ 8), markersize=6, c=dodgerblues[3], markerstrokewidth=0, label=false)
scatter!(s1, jitter(2, 28), mean.(rand_prs_cols[12:end] ./ 8), markersize=6,c=crimsons[3], markerstrokewidth=0, label=false)
scatter!(s1, jitter(3, 28), mean.(cfg_prs_cols[12:end] ./ 8), markersize=6,c=oranges[3], markerstrokewidth=0, label=false)
scatter!(s1, jitter(1, 9), mean.(conn_prs_cols[3:11] ./ 8), markersize=6,c=:dodgerblue4, markerstrokewidth=0, label=false)
scatter!(s1, jitter(2, 9), mean.(rand_prs_cols[3:11] ./ 8),markersize=6, c=:crimson, markerstrokewidth=0, label=false)
scatter!(s1, jitter(3, 9), mean.(cfg_prs_cols[3:11] ./ 8), markersize=6,c=:orange, markerstrokewidth=0, label=false, legend=false)
plot!(xticks=([1,2,3],["Conn","ER","CFG"]), size=(600,400),xtickfontsize=14,grid=false,  ytickfontsize=14, titlefontsize=16, leftmargin=5mm)



