function generate_cfg_network(D_in_sample, D_out_sample)
    in_degrees = D_in_sample
    out_degrees = D_out_sample


    if sum(in_degrees) != sum(out_degrees)
        error("The sum of in-degrees must equal the sum of out-degrees.")
    end

    # Create in/out stubs
    num_nodes = length(in_degrees)
    in_stubs = [i for (i, count) in enumerate(in_degrees) for _ in 1:count]
    out_stubs = [i for (i, count) in enumerate(out_degrees) for _ in 1:count]

    # Shuffle 
    shuffle!(in_stubs)
    shuffle!(out_stubs)

    # Pair stubs and avoid duplicate edges
    adjacency_matrix = zeros(Int, num_nodes, num_nodes)
    added_edges = Set{Tuple{Int, Int}}()  # A set to track existing edges so we don't duplicate

    for (src, dst) in zip(out_stubs, in_stubs)
        if (src, dst) ∉ added_edges
            adjacency_matrix[src, dst] += 1.0
            push!(added_edges, (src, dst))
        end
    end

    return adjacency_matrix
end

function node_degrees(W, node)
    """
    Calculate the in-degree and out-degree of a node in a weight matrix.

    Parameters:
    - W: Weight matrix (NxN), where W[i, j] is the weight from node j to node i.
    - node: The node index (integer).

    Returns:
    - in_degree: Number of nonzero incoming connections to the node.
    - out_degree: Number of nonzero outgoing connections from the node.
    """
    in_degree = count(!iszero, W[:, node])   # Count nonzero entries in the column
    out_degree = count(!iszero, W[node, :]) # Count nonzero entries in the row
    return in_degree, out_degree
end

function weighted_node_degrees(W, node)
    """
    Calculate the weighted in-degree and out-degree of a node in a weight matrix.

    Parameters:
    - W: Weight matrix (NxN), where W[i, j] is the weight from node j to node i.
    - node: The node index (integer).

    Returns:
    - weighted_in_degree: Sum of weights of incoming connections to the node.
    - weighted_out_degree: Sum of weights of outgoing connections from the node.
    """
    weighted_in_degree = sum(W[:, node])   # Sum weights in the column (incoming)
    weighted_out_degree = sum(W[node, :]) # Sum weights in the row (outgoing)
    return weighted_in_degree, weighted_out_degree
end

