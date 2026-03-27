
using Random, LinearAlgebra
using NaNMath, Plots

"""
generate_er_levels(K; N, S, q_levels)

Generate K Erdős–Rényi weight matrices for each level of self-recurrency q.
S = sparsity (fraction of zero entries).
q = fraction of neurons with self-recurrent weight.
"""
function generate_er_levels(K; N=100, S=0.985, q_levels=[0.0, 0.05, 0.15])

    p = 1 - S                 # connection probability
    levels = Dict()

    for q in q_levels
        mats = Matrix{Float64}[]

        for k in 1:K

            # generate ER matrix with zero diagonal 
            
            W = zeros(N, N)

            for i in 1:N, j in 1:N
                if i != j && rand() < p
                    W[i,j] = 2*rand() - 1  # weight in [-1,1]
                end
            end

            # impose desired self-recurrency 
            n_diag = round(Int, q * N)

            if n_diag > 0
                # indices of existing off-diagonal nonzeros
                off_inds = [(i,j) for i in 1:N, j in 1:N if i != j && W[i,j] != 0]

                shuffle!(off_inds)

                for idx in 1:min(n_diag, length(off_inds))
                    i,j = off_inds[idx]
                    W[i,i] = W[i,j]   # move weight to diagonal
                    W[i,j] = 0.0      # remove original
                end
            end

            push!(mats, W)
        end

        levels[q] = mats
    end

    return levels
end


function er_matrix(N, S)

    total_nonzeros = round(Int, (1 - S) * N^2)

    A = zeros(N, N)

    # choose exact locations
    idx = randperm(N^2)[1:total_nonzeros]

    for k in idx
        i = div(k-1, N) + 1
        j = mod(k-1, N) + 1
        A[i,j] = 2rand() - 1
    end

    return A
end


function reassign_to_diagonal(W::AbstractMatrix{<:Real}, q::Float64)

    N = size(W,1)

    offdiag = [(i,j) for i in 1:N, j in 1:N if i != j && W[i,j] != 0.0]

    num_to_move = round(Int, q*N)
    num_to_move = min(num_to_move, length(offdiag))

    sel = sample(offdiag, num_to_move; replace=false)

    # indices where diagonal is currently zero
    zero_diag = [i for i in 1:N if W[i,i] == 0.0]

    for (i,j) in sel

        # stop if no empty diagonal positions remain
        isempty(zero_diag) && break

        w = W[i,j]
        W[i,j] = 0.0

        # choose unused diagonal slot
        idx = rand(1:length(zero_diag))
        d = zero_diag[idx]

        W[d,d] = w

        # remove that slot from available list
        deleteat!(zero_diag, idx)
    end

    return W
end

num_levels = 10

matrices_base = [er_matrix(150, 0.985) for _ in 1:num_levels]
q_levels = [0.0, 0.05, 0.15, 0.20, 0.3]
matrices_05 = [reassign_to_diagonal(copy(W), 0.05) for W in matrices_base]
matrices_15 = [reassign_to_diagonal(copy(W), 0.15) for W in matrices_base]

spectral_radii = exp.(range(log(0.01), log(20.0), length=nsrs))

using Distributions


mle_base, D_base = analyze_ESNs_inputdriven(matrices_base, spectral_radii, [0.0])
mle_05, D_05 = analyze_ESNs_inputdriven(matrices_05, spectral_radii, [0.0])
mle_15, D_15 = analyze_ESNs_inputdriven(matrices_15, spectral_radii, [0.0])

three_green_colors = [:limegreen, :mediumseagreen, :seagreen]

p1 = plot(spectral_radii, mean(mle_01), label="q=0.02", fillalpha=0.3,  xscale=:log10,c=three_green_colors[1]) #, ribbon=NaNMath.std(mle_01)/sqrt(num_levels))
plot!(spectral_radii, mean(mle_05), label="q=0.05", fillalpha=0.3,  xscale=:log10,c=three_green_colors[2]) #, ribbon=NaNMath.std(mle_05)/sqrt(num_levels))
plot!(spectral_radii, mean(mle_15), label="q=0.15",fillalpha=0.3,   c=three_green_colors[3]) #, ribbon=NaNMath.std(mle_15)/sqrt(num_levels))
plot!(xlim=(0.05,20),legend=:bottomleft)

p2 = plot(spectral_radii, mean(D_01), fillalpha=0.3,  xscale=:log10, label="q=0.02", c=three_green_colors[1]) #, ribbon=NaNMath.std(D_01)/sqrt(num_levels))
plot!(spectral_radii, mean(D_05), fillalpha=0.3,  xscale=:log10, label="q=0.05", c=three_green_colors[2]) #, ribbon=NaNMath.std(D_05)/sqrt(num_levels))
plot!(spectral_radii, mean(D_15), fillalpha=0.3,  legend=false, label="q=0.15", c=three_green_colors[3]) #, ribbon=NaNMath.std(D_15)/sqrt(num_levels))
plot!(xlim=(0.05,20),)

p3 = plot(p1, p2, tickfontsize=16, legendfontsize=16,layout=(1,2), lw=4, size=(900,400), grid=false, title=["Max Lyapunov Exponent" "Lyapunov Dimension"], xlabel="Spectral Radius", ylabel=["MLE" "Dky"], margin=5mm)

