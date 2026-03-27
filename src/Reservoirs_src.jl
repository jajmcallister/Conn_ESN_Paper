module Reservoirs_src

using LinearAlgebra, SparseArrays
using OrdinaryDiffEq
using Statistics, Plots
using Arrow, DataFrames, CSV
using Dates, Random
using FileIO
using Interpolations

export compute_svd, participation_ratio, pca, valid_time_lorenz, valid_time_rossler, generate_arrow, create_reservoir, create_reservoir_mean_degree, create_reservoir_positive, create_input_weights, initialize_output_weights, update_reservoir_state, update_reservoir_state_linear, train_output_weights, lorenz!, save_data, training_lorenz_with_reservoir!, predicting_lorenz_with_reservoir!, predicting_lorenz_with_reservoir_sq!, old_create_dot_motion_data, create_LR_dot_motion_data_circle, generate_memory_target_data, create_training_and_testing_periods, calculate_nmse, square_even_indices!, lyapunov_exponent, lyapunov_exponent_from_history, generate_decisionmaking_data, generate_delay_decisionmaking_task_data

function compute_svd(W; return_leading=false)
    """
    Compute the Singular Value Decomposition (SVD) of an asymmetric matrix W.

    Arguments:
    - W::AbstractMatrix: Input matrix (asymmetric).

    Keyword Arguments:
    - return_leading::Bool: If true, return only the leading singular value and singular vectors.

    Returns:
    - U, Σ, Vt: The left singular vectors, singular values (diagonal matrix), and right singular vectors (transpose).
      OR
    - σ1, u1, v1: The leading singular value, left singular vector, and right singular vector (if return_leading=true).
    """
    # Perform SVD
    U, Σ, Vt = svd(W)

    if return_leading
        # Extract leading singular value and vectors
        σ1 = Σ[1]                # Leading singular value
        u1 = U[:, 1]             # Leading left singular vector
        v1 = Vt[1, :]            # Leading right singular vector
        return σ1, u1, v1
    else
        return U, Σ, Vt
    end
end


#defining lorenz system
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

function generate_arrow(name, data_path; force = false, compress = true)

    filename = joinpath(data_path, name)

    arrow_file = filename * ".arrow"

    if compress

        arrow_file = arrow_file * ".lz4"

    end

    csv_file = filename * ".csv"

    if !isfile(arrow_file) || force

        data = CSV.read(csv_file, DataFrame; normalizenames = true)

        if compress
            Arrow.write(arrow_file, data, compress = :lz4)
        else
            Arrow.write(arrow_file, data)
        end

    end

    return nothing

end

function save_data(variable, fold_name, name)
    # Get the current date/time
    current_datetime = now()
    # Format the current date/time
    formatted_datetime = Dates.format(current_datetime, "yyyy-mm-dd_HH-MM-SS")
    # Create the folder name
    folder_name = "$(name)" #_$(formatted_datetime)"
    # Create the folder
    mkdir(joinpath("C:/Users/B00955735/OneDrive - Ulster University/Documents/PhD/Code/Reservoirs_GraphTheory/Data files/$fold_name", folder_name))
    data_path = "C:/Users/B00955735/OneDrive - Ulster University/Documents/PhD/Code/Reservoirs_GraphTheory/Data files/$fold_name/$folder_name/"
    println("writing csv")
    CSV.write(data_path*name*".csv",DataFrame(variable,:auto),writeheader=true)
    println("writing arrow")
    generate_arrow(name,data_path)
    println("deleting csv")
    rm(data_path*name*".csv")
end

function read_arrow_data(path_name)
    m = Matrix(DataFrame(Arrow.Table(path_name)))
    return m
end

function create_reservoir(size::Int, sparsity::Float64, spectral_radius::Float64)
    # Create a sparse, random matrix for the reservoir
    smat = sprand(size,size,sparsity)

    values=smat.nzval

    values=[2*v-1 for v in values]

    indices=findnz(smat)

    reservoir =sparse(indices[1],indices[2],values,size,size)
    # Adjust spectral radius
    eigenvalues = eigvals(Matrix(reservoir))
    max_eigenvalue = maximum(abs.(eigenvalues))
    reservoir *= spectral_radius / max_eigenvalue
    return reservoir
end

function create_reservoir_mean_degree(size,mean_degree,spectral_radius)
    edge_probability = mean_degree/size
    decision_samples=rand(size,size)
    reservoir_weight_matrix = zeros(size,size)

 

    for i in 1:size

        for j in 1:size

            if decision_samples[i,j]<=edge_probability

                #weights uniform sample ∈ [-1,1]

                reservoir_weight_matrix[i,j]=2*rand()-1

            end

        end

    end

 

    #get current spectral radius

    res_ρ = maximum(abs.(eigvals(reservoir_weight_matrix)))

    #scale to desired spectral radius

    reservoir_weight_matrix .*= spectral_radius/res_ρ
    return reservoir_weight_matrix
end

function create_reservoir_positive(size::Int, sparsity::Float64, spectral_radius::Float64)
    # Create a sparse, random matrix for the reservoir
    smat = sprand(size,size,sparsity)

    values=smat.nzval

    indices=findnz(smat)

    reservoir =sparse(indices[1],indices[2],values,size,size)
    # Adjust spectral radius
    eigenvalues = eigvals(Matrix(reservoir))
    max_eigenvalue = maximum(abs.(eigenvalues))
    reservoir *= spectral_radius / max_eigenvalue
    return reservoir
end

function create_input_weights(input_size::Int, reservoir_size::Int, input_scaling::Float64)
    return (rand(reservoir_size, input_size) .* 2 .- 1)*input_scaling # Random values between -1 and 1 scaled by input scaling
end

function initialize_output_weights(reservoir_size::Int, output_size::Int)
    return zeros(output_size, reservoir_size)
end

function update_reservoir_state(reservoir, input_weights, current_state, input, leak_rate)
    return (1-leak_rate)*current_state + leak_rate*tanh.(reservoir * current_state + input_weights * input)
end

function update_reservoir_state_linear(reservoir, input_weights, current_state, input, leak_rate)
    return (1-leak_rate)*current_state + leak_rate*(reservoir * current_state + input_weights * input)
end


function train_output_weights(reservoir_states, target_outputs, regularization_coefficient)
    # Using ridge regression
    x = Matrix::Any
    try
        x = (reservoir_states * reservoir_states' + regularization_coefficient * I) \ (reservoir_states * target_outputs')
        
        # Check for Inf or NaN values in the result
        if any(isinf, x) || any(isnan, x)
            x = zeros(size(target_outputs, 1),size(reservoir_states, 1))
        end
    catch e
        x = zeros(size(target_outputs, 1),size(reservoir_states, 1))
    end

    return x
end


function training_lorenz_with_reservoir!(du, u, p, t)
    reservoir, input_weights, leak_rate, train_lorenz_data = p
    itp = interpolate(train_lorenz_data, BSpline(Linear()))
    data_at_t = itp(1:3, t)  # Interpolate over the first three rows
    res_size = size(reservoir)[1]
    du[1:res_size] = (1-leak_rate)*u[1:res_size] + leak_rate .* tanh.(reservoir * u[1:res_size] + input_weights * data_at_t[1:3]) - u[1:res_size]
end


function predicting_lorenz_with_reservoir!(du, u, p2, t)
    reservoir, input_weights, output_weights, leak_rate, test_lorenz_data = p2
    res_size=size(reservoir,1)
    itp = interpolate(test_lorenz_data, BSpline(Linear()))
    data_at_t = itp(1:3, t)  # Interpolate over the first three rows
    if t < 100
        du[1:res_size] = (1 - leak_rate) .* u[1:res_size] + leak_rate .* tanh.(reservoir * u[1:res_size] + input_weights * data_at_t[1:3]) - u[1:res_size]
    else
        last_output = output_weights'*u[1:res_size]
        du[1:res_size] = (1 - leak_rate) .* u[1:res_size] + leak_rate .* tanh.(reservoir * u[1:res_size] + input_weights * last_output) - u[1:res_size]
    end
end

function square_even_indices!(matrix)
    # Loop through every column
    for col in 1:size(matrix, 2)
        # Loop through every even row
        for row in 2:2:size(matrix, 1)
            matrix[row, col] = matrix[row, col]^2
        end
    end
end

function predicting_lorenz_with_reservoir_sq!(du, u, p2, t)
    reservoir, input_weights, output_weights, leak_rate, test_lorenz_data = p2
    res_size=size(reservoir,1)
    itp = interpolate(test_lorenz_data, BSpline(Linear()))
    data_at_t = itp(1:3, t)  # Interpolate over the first three rows
    if t < 100
        du[1:res_size] = (1 - leak_rate) .* u[1:res_size] + leak_rate .* tanh.(reservoir * u[1:res_size] + input_weights * data_at_t[1:3]) - u[1:res_size]
    else
        square_even_indices!(u[1:res_size])
        last_output = output_weights'*u[1:res_size]
        du[1:res_size] = (1 - leak_rate) .* u[1:res_size] + leak_rate .* tanh.(reservoir * u[1:res_size] + input_weights * last_output) - u[1:res_size]
    end
end

function create_training_and_testing_periods(data, train_length, test_length)
    train_length = Int(train_length)
    train_period = data[:,1:train_length]
    testing_period = data[:,train_length+1:end]
    test_periods = []
    x = round(Int, size(testing_period,2)/test_length - 1)
    for i in 1:x 
        if (i+1)*test_length < size(testing_period,2)
            push!(test_periods, data[:,i*test_length:(i+1)*test_length])
        end
    end
    return train_period, test_periods
end

function calculate_nmse(predictions, data)
    l2_norm = sqrt.(sum((data .- predictions).^2,dims=1))

    n = size(data,2)

    norm_error = l2_norm/ sqrt((1/n).*sum(sum(data.^2,dims=1)))

    return norm_error
end


function valid_time_lorenz(threshold, predictions, target, dt)
    dt = 0.02
    predictions = predictions[:,100:end]
    target = target[:,100:end]
    norm_error = calculate_nmse(predictions, target)
    valid_time_index = findfirst(x->(x>threshold), norm_error)
    lorenz_maxlyap = 0.9056;
    # Check if valid_time_index is nothing
    if valid_time_index === nothing
        return size(predictions,2), size(predictions,2)*dt/lorenz_maxlyap
    else
        valid_lt = dt*valid_time_index[2]*(1/lorenz_maxlyap)
        return valid_time_index[2], valid_lt
    end
end

function valid_time_rossler(threshold, predictions, target, dt)
    predictions = predictions[:,100:end]
    target = target[:,100:end]
    norm_error = calculate_nmse(predictions, target)
    valid_time_index = findfirst(x->(x>threshold), norm_error)
    rossler_maxlyap = 0.09;
    # Check if valid_time_index is nothing
    if valid_time_index === nothing
        return size(predictions,2), size(predictions,2)*dt/rossler_maxlyap
    else
        valid_lt = dt*valid_time_index[2]*(1/rossler_maxlyap)
        return valid_time_index[2], valid_lt
    end
end

function valid_time_oscillator(threshold, predictions, target, dt)
    predictions = predictions[:,100:end]
    target = target[:,100:end]
    norm_error = calculate_nmse(predictions, target)
    valid_time_index = findfirst(x->(x>threshold), norm_error)
    # Check if valid_time_index is nothing
    if valid_time_index === nothing
        return size(predictions,2), size(predictions,2)*dt
    else
        valid_lt = dt*valid_time_index[2]*(1)
        return valid_time_index[2], valid_lt
    end
end

function valid_time_lotka(threshold, predictions, target, dt)
    predictions = predictions[:,100:end]
    target = target[:,100:end]
    norm_error = calculate_nmse(predictions, target)
    valid_time_index = findfirst(x->(x>threshold), norm_error)
    # Check if valid_time_index is nothing
    if valid_time_index === nothing
        return size(predictions,2), size(predictions,2)*dt
    else
        valid_lt = dt*valid_time_index[2]*(1)
        return valid_time_index[2], valid_lt
    end
end


function dijkstra(graph, start)
    n = size(graph, 1)
    dist = fill(Inf, n)
    dist[start] = 0
    visited = falses(n)
    
    while true
        # Find the vertex with the smallest tentative distance
        min_dist = Inf
        min_vertex = -1
        for v = 1:n
            if !visited[v] && dist[v] < min_dist
                min_dist = dist[v]
                min_vertex = v
            end
        end
        
        # If all vertices are visited or unreachable, break the loop
        if min_vertex == -1
            break
        end
        
        visited[min_vertex] = true
        
        # Update distances to neighbors of min_vertex
        for v = 1:n
            if graph[min_vertex, v] > 0  # Check for adjacency
                alt = dist[min_vertex] + graph[min_vertex, v]
                if alt < dist[v]
                    dist[v] = alt
                end
            end
        end
    end
    
    return dist
end

function average_path_length(matrix)
    n, _ = size(matrix)
    total_path_length = 0
    num_paths = 0
    # wmatrix = 1 ./ matrix
    wmatrix = deepcopy(matrix)
    wmatrix .!= 0
    
    for start = 1:n
        shortest_distances = dijkstra(wmatrix, start)
        for dist in shortest_distances
            if dist != 0 && dist != Inf
                total_path_length += dist
                num_paths += 1
            end
        end
    end
    
    return total_path_length / num_paths
end

function equivalent_random_matrix(weight_matrix, w)

    vertices = size(weight_matrix, 1)
    edges = count(!iszero, weight_matrix)

    # Initialize a matrix with zeros
    m = zeros(vertices, vertices)
    num_edges = 0 
    while num_edges < edges
        u, v = rand(1:vertices, 2)
        # if u != v && m[u,v]==0
        if m[u,v]==0
            m[u,v]=w
            num_edges+=1
        end
    end
    l = size(m,1)
    m[1:l+1:end] .= 0
    return m
end

function directed_weighted_clustering_coefficient(wmatrix)
    n = size(wmatrix, 1)
    clustering_coefficients = zeros(n)
    
    for i = 1:n
        neighbors_out = findall(x -> x > 0, wmatrix[i, :])
        neighbors_in = findall(x -> x > 0, wmatrix[:, i])
        common_neighbors = intersect(neighbors_in, neighbors_out)
        
        if length(common_neighbors) > 1
            triad_strength_sum = 0.0
            triads = 0
            
            for j in common_neighbors
                for k in common_neighbors
                    if wmatrix[i, j] > 0 && wmatrix[j, k] > 0 && wmatrix[k, i] > 0
                        # Geometric mean of the triangle weights
                        triad_strength = (wmatrix[i, j] * wmatrix[j, k] * wmatrix[k, i])^(1/3)
                        triad_strength_sum += triad_strength
                        triads += 1
                    end
                end
            end
            
            if triads > 0
                clustering_coefficients[i] = triad_strength_sum / triads
            else
                clustering_coefficients[i] = 0
            end
        else
            clustering_coefficients[i] = 0
        end
    end
    
    return mean(clustering_coefficients)
end

function calculate_small_world(res)
    matrix = Matrix(res)
    mean = Statistics.mean(matrix[matrix.>0])
    l = average_path_length(matrix)
    clust_coeff = directed_weighted_clustering_coefficient(matrix)
    rand_equiv_matrix = equivalent_random_matrix(matrix, mean)
    clust_coeff_rand_equiv = directed_weighted_clustering_coefficient(rand_equiv_matrix)
    L_rand_equiv = average_path_length(rand_equiv_matrix)
    if clust_coeff==0
        sigma = 0
    else
        sigma = (clust_coeff/clust_coeff_rand_equiv)/(l/L_rand_equiv)
    end
    
    return sigma
end

#defining lorenz system
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

function get_node_degree(matrix)
    adj = Matrix(copy(matrix))
    adj[abs.(adj).<0.001].=0.
    edges = findall(!=(0.0),adj)
    adj[edges].=1.0

    return sum(adj,dims=1)
end

function generate_memory_target_data(data, output_size::Int)
    target_data = zeros(length(data), output_size)
    for tau in 1:output_size
        target_data[tau+1:end, tau] = data[1:end-tau]
    end
    return target_data
end

function create_reservoir_normal(size::Int, sparsity::Float64, spectral_radius::Float64)
    # Create a sparse, random matrix for the reservoir
    smat = sprand(size, size, sparsity)


    values = smat.nzval

    # Replace uniform distribution with normal distribution
    values = randn(length(values))

    indices = findnz(smat)

    reservoir = sparse(indices[1], indices[2], values, size, size)

    # Adjust spectral radius
    eigenvalues = eigvals(Matrix(reservoir))
    max_eigenvalue = maximum(abs.(eigenvalues))
    reservoir *= spectral_radius / max_eigenvalue
    return reservoir
end

function create_reservoir_cauchy(size::Int, sparsity::Float64, spectral_radius::Float64)
    # Create a sparse, random matrix for the reservoir
    smat = sprand(size, size, sparsity)

    values = smat.nzval

    # Replace uniform distribution with Cauchy distribution
    values = rand(Cauchy(0, 1), length(values))

    indices = findnz(smat)

    reservoir = sparse(indices[1], indices[2], values, size, size)

    # Adjust spectral radius
    eigenvalues = eigvals(Matrix(reservoir))
    max_eigenvalue = maximum(abs.(eigenvalues))
    reservoir *= spectral_radius / max_eigenvalue
    return reservoir
end

# Function to compute the largest Lyapunov exponent
function lyapunov_exponent(W, Win, u, input_sequence, num_steps)
    size_reservoir = size(W, 1)
    
    # Initial reservoir state
    x = randn(size_reservoir)
    
    # Initialize perturbation vector (random small vector)
    delta_x = randn(size_reservoir) * 1e-8
    delta_x /= norm(delta_x)
    
    sum_log_divergence = 0.0

    for t in 1:num_steps
        input = input_sequence[t]

        # Update the reservoir states
        x_next = reservoir_update(W, x, u, Win, input)

        # Update the perturbed state
        perturbed_x = x + delta_x
        perturbed_x_next = reservoir_update(W, perturbed_x, u, Win, input)
        
        # Compute the difference between the trajectories
        delta_x_new = perturbed_x_next - x_next

        # Compute the divergence factor (growth of the perturbation)
        divergence = norm(delta_x_new)
        sum_log_divergence += log(divergence)

        # Renormalize the perturbation
        delta_x = delta_x_new / divergence

        # Update the state
        x = x_next
    end
    
    # Calculate the Lyapunov exponent (average rate of divergence)
    return sum_log_divergence / num_steps
end

# Function to compute the largest Lyapunov exponent from reservoir state history
function lyapunov_exponent_from_history(X::Matrix{Float64}, δ0::Float64, num_steps::Int)
    # X: Reservoir state history (each column is a state vector at time t)
    # δ0: Initial perturbation size
    # num_steps: Number of steps to compute the Lyapunov exponent

    size_reservoir, T = size(X)  # T is the number of time steps

    # Choose two nearby initial states
    x0 = X[:, 1]  # Initial state
    δ = randn(size_reservoir) * δ0  # Small initial perturbation
    x0_perturbed = x0 + δ  # Perturbed initial state

    # Initialize the sum of log divergences
    sum_log_divergence = 0.0

    for t in 1:num_steps-1
        # Get original and perturbed state at time t
        xt = X[:, t]           # Original state
        xt_perturbed = X[:, t+1]  # Next state

        # Calculate the divergence (difference between trajectories)
        δ_new = xt_perturbed - xt

        # Calculate divergence factor (growth of perturbation)
        divergence = norm(δ_new)
        if divergence == 0.0
            continue  # Avoid division by zero or log of zero
        end
        sum_log_divergence += log(divergence / norm(δ))

        # Renormalize the perturbation
        δ = δ_new / divergence
    end

    # Compute the Lyapunov exponent (average rate of divergence)
    return sum_log_divergence / num_steps
end

function generate_decisionmaking_data(num_samples::Int, switch_interval::Int, bias)
    # Create random inputs for two input neurons
    input_data = zeros(Float64, 2, num_samples)
    target_output = zeros(Float64, num_samples)
    
    # Initialize current dominant neuron (true for neuron 1, false for neuron 2)
    current_high = rand(Bool)
    
    for i in 1:num_samples
        # Every `switch_interval` steps, randomly decide whether to switch
        if i % switch_interval == 1 && i > 1
            if rand() < 0.5  # 50% chance to switch
                current_high = !current_high
            end
        end
        
        # Generate random inputs for both neurons
        input_data[:, i] = randn(2)
        
        # Add a small bias to the dominant neuron
        if current_high
            input_data[1, i] += bias  # Neuron 1 gets a small bias
            target_output[i] = 1.0    # Target output is +1
        else
            input_data[2, i] += bias  # Neuron 2 gets a small bias
            target_output[i] = -1.0   # Target output is -1
        end
    end
    
    return input_data, target_output
end

function generate_delay_decisionmaking_task_data(num_trials, stim_duration, resp_duration, variance)
    # Preallocate storage
    input_neuron1 = []  # Sensory evidence
    input_neuron2 = []  # Response period indicator
    outputs = []        # Expected output

    for _ in 1:num_trials
        # Randomly assign a mean direction (-1 or +1) for the trial
        trial_mean = rand([-1, 1]) * 0.5  # ±0.5 mean motion direction

        # Generate noisy sensory evidence for the stimulus presentation period
        stimulus = randn(stim_duration) .* sqrt(variance) .+ trial_mean  # Gaussian random variable
        
        # Response period indicator (0 during stimulus, 1 during response period)
        response_indicator = vcat(zeros(stim_duration), ones(resp_duration))
        
        # Expected output during fixation/stimulus presentation is 0, during response is ±1
        trial_output = vcat(zeros(stim_duration), fill(sign(trial_mean), resp_duration))
        
        # Concatenate inputs and outputs for the trial
        input_neuron1 = vcat(input_neuron1, stimulus, zeros(resp_duration))
        input_neuron2 = vcat(input_neuron2, response_indicator)
        outputs = vcat(outputs, trial_output)
    end
    
    # Combine inputs into a matrix
    inputs = hcat(input_neuron1, input_neuron2)
    
    return inputs', Float64.(outputs)
end

# PCA function from scratch
function pca(X::Matrix; num_components=nothing)
    # Center the data (subtract mean of each row)
    X_centered = X .- mean(X, dims=2)
    
    # Compute covariance matrix
    cov_matrix = (X_centered * X_centered') / (size(X, 2) - 1)
    
    # Perform eigendecomposition of the covariance matrix
    eigenvalues, eigenvectors = eigen(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = sortperm(eigenvalues, rev=true)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Optionally, keep only the top num_components
    if num_components !== nothing
        eigenvectors = eigenvectors[:, 1:num_components]
        eigenvalues = eigenvalues[1:num_components]
    end
    
    # Project data onto the principal components
    X_pca = eigenvectors' * X_centered
    
    return X_pca, eigenvalues, eigenvectors
end

function participation_ratio(eigenvalues)
    S1 = sum(eigenvalues)
    S2 = sum(eigenvalues .^ 2)
    return S1^2 / S2
end

end

