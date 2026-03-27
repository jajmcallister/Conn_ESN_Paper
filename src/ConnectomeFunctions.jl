using .Reservoirs_src

module ConnectomeFunctions

using Random, LinearAlgebra, Statistics, StatsBase
using LightGraphs, Graphs, GraphPlot, SimpleGraphs, GraphRecipes, SimpleWeightedGraphs, NetworkLayout
using Plots

export count_recurrent, adjust_ratio_EI, scale_matrices, eigenspectral_analysis, compute_sparsity, convert_to_real_vector, flatten_nested_vectors_into_matrix, negate_half_nonzero, shuffle_matrix, count_isolated_nodes, remove_isolated_nodes, plot_connectivity_network, weight_to_adjacency, randomise_nonzero_elements


function compute_sparsity(matrix)
    num_nonzero = count(!iszero, matrix)
    total_elements = length(matrix)
    sparsity = num_nonzero / total_elements
    return sparsity
end

function convert_to_real_vector(vector_any::Vector{Any})
    # Initialize an empty Vector{Real}
    vector_real = Vector{Real}(undef, length(vector_any))

    # Iterate over each element in the Vector{Any}
    for i in 1:length(vector_any)
        # Try to convert the element to a Real value
        try
            vector_real[i] = convert(Real, vector_any[i])
        catch
            error("Unable to convert element at index $i to Real")
        end
    end

    return vector_real
end

# Recursive function to flatten nested vectors
function flatten_nested_vectors_into_matrix(vec)
    flat_vec = []
    for elem in vec
        if elem isa Vector
            append!(flat_vec, flatten_nested_vectors(elem))
        else
            push!(flat_vec, elem)
        end
    end
    M = reshape(flat_vec, length(flat_V) ÷ length(vec), length(vec))
    return M'
end

function negate_half_nonzero(matrix)
    # Get indices of nonzero elements
    nonzero_indices = findall(!iszero, matrix)
    
    # Randomly shuffle the nonzero indices
    shuffled_indices = randperm(length(nonzero_indices))
    
    # Select the first half of shuffled indices
    half_indices = shuffled_indices[1:length(shuffled_indices) ÷ 2]
    
    # Negate the corresponding elements in the original matrix
    for idx in half_indices
        matrix[nonzero_indices[idx]] *= -1
    end
    
    return matrix
end



function count_isolated_nodes(weight_matrix)
    num_nodes = size(weight_matrix, 1)
    isolated_nodes = 0
    
    for node in 1:num_nodes
        if sum(weight_matrix[node, :]) == 0 && sum(weight_matrix[:, node]) == 0
            isolated_nodes += 1
        end
    end
    
    return isolated_nodes
end

function remove_isolated_nodes(matrix)
    num_nodes = size(matrix, 1)
    
    isolated_nodes = Int[]
    
    # Identify isolated nodes
    for node in 1:num_nodes
        if sum(matrix[node, :]) == 0 && sum(matrix[:, node]) == 0
            push!(isolated_nodes, node)
        end
    end

    # Remove isolated nodes
    for node in reverse(isolated_nodes)
        matrix = delete_row_and_column!(matrix, node)
    end

    return matrix
end

# Example function for deleting row and column in-place
function delete_row_and_column!(matrix, i)
    # Delete ith row
    matrix = matrix[setdiff(1:end, i), :]
    
    # Delete ith column
    matrix = matrix[:, setdiff(1:end, i)]
    
    return matrix
end

function plot_connectivity_network(weight_matrix)
    num_nodes = size(weight_matrix, 1)
    
    # Create a directed graph
    graph = SimpleWeightedDiGraph(num_nodes)
    
    # Add directed edges based on the weight matrix
    for i in 1:num_nodes
        for j in 1:num_nodes
            if weight_matrix[i, j] != 0
                SimpleWeightedGraphs.add_edge!(graph, i, j)
            end
        end
    end

    # Calculate a scaling factor based on the maximum weight in the matrix
    scaling_factor = maximum(weight_matrix) / 10

    # Use the scaling factor to adjust the node distance
    node_distance = 50 * scaling_factor

    # Define edge widths based on weights
    graphplot(graph, edgewidth=weight_matrix/4, curvature_scalar=0.0, method=:spring, size=(2000,2000), node_distance=node_distance, node_shape=:circle)
end

# Function to get eigenspectrum and PR
function eigenspectral_analysis(M)
    λ, v = eigen(M)

    # Ensure eigenvalues are real
    λ = abs.(real(λ))
    
    # eigenvalues in descending order
    idx = sortperm(λ, rev=true)
    λ_sorted = λ[idx]
    
    #participation ratio
    PR = sum(λ)^2 / sum(λ.^2)
    
    return λ_sorted, PR
end



function weight_to_adjacency(weight_matrix, threshold)
    n = size(weight_matrix, 1)
    adjacency_matrix = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if weight_matrix[i, j] >= threshold
                adjacency_matrix[i, j] = 1
            end
        end
    end
    return adjacency_matrix
end



function create_reservoir_EI(size::Int, sparsity::Float64, spectral_radius::Float64)

    # Create a sparse, random matrix for the reservoir
    smat = sprand(size, size, sparsity)
    values = smat.nzval

    # Create 4:1 ratio of positive to negative values
    num_positive = Int(floor(0.8 * length(values)))  # 80% positive
    num_negative = length(values) - num_positive     # 20% negative

    # Generate the new values with the desired ratio
    shuffled_indices = shuffle(1:length(values))
    positive_indices = shuffled_indices[1:num_positive]
    negative_indices = shuffled_indices[num_positive+1:end]

    values[positive_indices] .= rand(num_positive) # Positive values between -1 and 1
    values[negative_indices] .= -(rand(num_negative))  # Negative values between -1 and 1

    # Ensure values are within the desired range [-1, 1]
    values .= clamp.(values, -1, 1)

    # Reconstruct the sparse matrix with the modified values
    indices = findnz(smat)
    reservoir = sparse(indices[1], indices[2], values, size, size)

    # Adjust spectral radius
    eigenvalues = eigvals(Matrix(reservoir))
    max_eigenvalue = maximum(abs.(eigenvalues))
    reservoir *= spectral_radius / max_eigenvalue

    return reservoir
end


function adjust_ratio_EI(matrix::AbstractMatrix, pos_ratio,neg_ratio)
    # Extract nonzero values and their indices
    nonzero_indices = findall(!iszero, matrix)
    nonzero_values = matrix[nonzero_indices]

    # Count the current positive and negative values
    num_positive = count(x -> x > 0, nonzero_values)
    num_negative = count(x -> x < 0, nonzero_values)

    total_nonzero = length(nonzero_values)
    desired_positive = Int(floor(total_nonzero * pos_ratio / (pos_ratio + neg_ratio)))
    desired_negative = total_nonzero - desired_positive

    # Determine how many values need to be negated
    if num_positive > desired_positive
        # More positives than desired, negate some positives
        num_to_negate = num_positive - desired_positive
        indices_to_negate = findall(x -> x > 0, nonzero_values)
    else
        # More negatives than desired, negate some negatives
        num_to_negate = num_negative - desired_negative
        indices_to_negate = findall(x -> x < 0, nonzero_values)
    end

    # Randomly select and negate the required number of values
    if num_to_negate > 0
        selected_indices = randperm(length(indices_to_negate))[1:num_to_negate]
        for idx in selected_indices
            nonzero_values[indices_to_negate[idx]] *= -1
        end
    end

    # Assign the adjusted values back to the matrix
    matrix[nonzero_indices] .= nonzero_values

    return matrix
end


function scale_matrices(matrices, desired_radius::Float64)
    # Iterate over each matrix in the vector
    matrixxes = []
    for i in 1:length(matrices)
        matrix = matrices[i]
        
        # Ensure the matrix is of type Float64
        matrix = Matrix{Float64}(matrix)
        
        # Compute the spectral radius
        eigenvalues = eigvals(matrix)
        max_eigenvalue = maximum(abs.(eigenvalues))
        spectral_radius = max_eigenvalue
        
        # Calculate the scaling factor
        scaling_factor = desired_radius / spectral_radius
        
        # Scale the matrix element-wise
        matrixx = matrix .* scaling_factor
        push!(matrixxes, matrixx)
    end
    
    return matrixxes
end


function count_self_loops(W)
    return sum(diag(W) .!= 0)
end

# Function to count 2-cycles
function count_2_cycles(W)
    A = W .!= 0  # adjacency matrix
    return sum(A * A') - tr(A * A')
end

# Function to count 3-cycles
function count_3_cycles(W)
    A = W .!= 0  # adjacency matrix
    return sum(diag(A * A * A))
end

function count_recurrent(W)
    self_count = count_self_loops(W)
    count_2 = count_2_cycles(W)
    count_3 = count_3_cycles(W)
    total = self_count #+count_2+count_3
    return total
end

function shuffle_matrix(M, level)
    # Ensure the level is between 0 and 1
    level = clamp(level, 0.0, 1.0)
    
    # Flatten the matrix to a vector for easier shuffling
    flat_M = vec(M)
    
    # Number of elements to shuffle
    num_elements = length(flat_M)
    num_to_shuffle = round(Int, num_elements * level)
    
    # Generate random indices to shuffle
    indices = randperm(num_elements)[1:num_to_shuffle]
    
    # Extract the elements to be shuffled
    elements_to_shuffle = flat_M[indices]
    
    # Shuffle the extracted elements
    shuffle!(elements_to_shuffle)
    
    # Place the shuffled elements back in the original positions
    flat_M[indices] = elements_to_shuffle
    
    # Reshape the vector back to the original matrix shape
    reshuffled_M = reshape(flat_M, size(M))
    
    return reshuffled_M
end

function convert_to_float_matrix(matrix)
    rows, cols = size(matrix)
    float_matrix = Array{Float64}(undef, rows, cols)
    
    for i in 1:rows
        for j in 1:cols
            float_matrix[i, j] = ismissing(matrix[i, j]) ? missing : parse(Float64, matrix[i, j])
        end
    end
    
    return float_matrix
end

function randomise_nonzero_elements(M)
    # Create a copy of the input matrix to modify
    M_new = copy(M)
    M_new = float(M_new)

    # Iterate through each element of the matrix
    for i in 1:size(M, 1)
        for j in 1:size(M, 2)
            if M[i, j] != 0
                M_new[i, j] = 2*rand()-1  # Assign a random value between -1 and 1
            end
        end
    end

    return M_new
end



end