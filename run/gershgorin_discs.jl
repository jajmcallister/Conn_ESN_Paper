using LinearAlgebra
using Random
using StatsBase
using Plots.PlotMeasures
using Statistics
using Plots

# Theory

function relative_theory_sr_gersh_uniform(N,S,q,f)
    k = (1-S)*N - q
    kf = max(0.0, (1-S)*N*(1-f) - q)

    # diagonal and bulk contributions
    diag0 = (q*N)/(q*N+1)
    diagf = (q*N*(1-f))/(q*N*(1-f)+1)

    gersh0 = k/2  # this is gershgorin radius theory (using absolute row sums)
    gershf = kf/2

    rho0 = diag0+gersh0
    rhof = diagf+gershf

    return rhof / rho0
end

function relative_theory_sr_gersh_gaussian(N,S,q,f)
    k = (1-S)*N - q
    kf = max(0.0, (1-S)*N*(1-f) - q)

    # effective diagonal counts
    nq0 = max(N*q, 1)
    nqf = max(N*q*(1-f), 1)

    # diagonal contributions
    diag0 = sqrt(2*log(nq0))
    diagf = sqrt(2*log(nqf))

    gersh0 =  k*sqrt(2/pi) # this is gershgorin radius theory (using absolute row sums)
    gershf =  kf*sqrt(2/pi)

    rho0 = diag0+gersh0
    rhof = diagf+gershf

    return rhof / rho0
end

function relative_theory_sr_max_uniform(N,S,q,f)
    k = (1-S)*N - q
    kf = max(0.0, (1-S)*N*(1-f) - q) 

    # diagonal and bulk contributions
    diag0 = (q*N)/(q*N+1)
    diagf = (q*N*(1-f))/(q*N*(1-f)+1)

    bulk0 = sqrt(k/3) # this is circular law theory
    bulkf = sqrt(kf/3)

    rho0 = max(diag0, bulk0)
    rhof = max(diagf, bulkf)

    return rhof / rho0
end

function relative_theory_sr_max_gaussian(N,S,q,f)
    k = (1-S)*N - q
    kf = max(0.0, (1-S)*N*(1-f) - q)

    # effective diagonal counts
    nq0 = max(N*q, 1)
    nqf = max(N*q*(1-f), 1)

    # diagonal contributions
    diag0 = sqrt(2*log(nq0))
    diagf = sqrt(2*log(nqf))

    # bulk contributions
    bulk0 = sqrt(k)
    bulkf = sqrt(kf)

    rho0 = max(diag0, bulk0)
    rhof = max(diagf, bulkf)

    return rhof / rho0
end


# Empirical

function er_matrix_uniform(N, S)

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

function er_matrix_gaussian(N, S)

    total_nonzeros = round(Int, (1 - S) * N^2)

    A = zeros(N, N)

    # choose exact locations
    idx = randperm(N^2)[1:total_nonzeros]

    for k in idx
        i = div(k-1, N) + 1
        j = mod(k-1, N) + 1
        A[i,j] = randn()
    end

    return A
end

spec_rad(A) = maximum(abs.(eigvals(A)))

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


function prune_matrix(W, f)

    N = size(W,1)
    n_prune = round(Int, f*N)

    keep = setdiff(1:N, randperm(N)[1:n_prune])

    return W[keep, keep]

end

#################################################


using Base.Threads

Ns=[100,200,300]
sparsities = [0.8,0.9,0.985]


uniform_empirical = Matrix{Any}(undef, length(Ns), length(sparsities))
uniform_theory = Matrix{Any}(undef, length(Ns), length(sparsities))
gaussian_empirical = Matrix{Any}(undef, length(Ns), length(sparsities))
gaussian_theory = Matrix{Any}(undef, length(Ns), length(sparsities))


n_trials = 10

S = 0.985

qs = range(0.0, 1.0, length=100)
fs = range(0.0, 0.99, length=100)



Threads.@threads for idx in 1:(length(Ns)*length(sparsities))

    in = div(idx-1, length(sparsities)) + 1
    js = mod(idx-1, length(sparsities)) + 1

    N = Ns[in]
    S = sparsities[js]

    R_uniform = zeros(length(qs), length(fs))
    R_gaussian = zeros(length(qs), length(fs))

    for (qi,q) in enumerate(qs)
        for trial in 1:n_trials

            A_base_uniform = er_matrix_uniform(N, S)
            A_base_gaussian = er_matrix_gaussian(N, S)

            A_uniform = reassign_to_diagonal(copy(A_base_uniform), q)
            A_gaussian = reassign_to_diagonal(copy(A_base_gaussian), q)

            rho0_uniform = spec_rad(A_uniform)
            rho0_gaussian = spec_rad(A_gaussian)

            for (fi,f) in enumerate(fs)

                Ap_uniform = prune_matrix(A_uniform, f)
                Ap_gaussian = prune_matrix(A_gaussian, f)

                rho_uniform = spec_rad(Ap_uniform)
                rho_gaussian = spec_rad(Ap_gaussian)

                R_uniform[qi,fi] += rho_uniform / rho0_uniform
                R_gaussian[qi,fi] += rho_gaussian / rho0_gaussian
            end
        end
    end

    R_uniform ./= n_trials
    R_gaussian ./= n_trials

    Z_uniform_max = zeros(length(qs), length(fs))
    Z_gaussian_max = zeros(length(qs), length(fs))

    for (i,q) in enumerate(qs), (j,f) in enumerate(fs)
        Z_uniform_max[i,j] = relative_theory_sr_max_uniform(N,S,q,f)
        Z_gaussian_max[i,j] = relative_theory_sr_max_gaussian(N,S,q,f)
    end

    uniform_empirical[in, js] = R_uniform
    uniform_theory[in, js] = Z_uniform_max
    gaussian_empirical[in, js] = R_gaussian
    gaussian_theory[in, js] = Z_gaussian_max
end


uniform_empiricals = [heatmap(fs, qs, uniform_empirical[i,j], xlabel="Fraction pruned f", ylabel="Diagonal fraction q", clim=(0,1), title="Empirical N=$(Ns[i]) S=$(sparsities[j])", c=:viridis) for i in 1:length(Ns), j in 1:length(sparsities)]
uniform_theories = [heatmap(fs, qs, uniform_theory[i,j], xlabel="Fraction pruned f", ylabel="Diagonal fraction q", clim=(0,1), title="Theory N=$(Ns[i]) S=$(sparsities[j])", c=:viridis) for i in 1:length(Ns), j in 1:length(sparsities)]
gaussian_empiricals = [heatmap(fs, qs, gaussian_empirical[i,j], xlabel="Fraction pruned f", ylabel="Diagonal fraction q", clim=(0,1), title="Empirical N=$(Ns[i]) S=$(sparsities[j])", c=:viridis) for i in 1:length(Ns), j in 1:length(sparsities)]
gaussian_theories = [heatmap(fs, qs, gaussian_theory[i,j], xlabel="Fraction pruned f", ylabel="Diagonal fraction q", clim=(0,1), title="Theory N=$(Ns[i]) S=$(sparsities[j])", c=:viridis) for i in 1:length(Ns), j in 1:length(sparsities)]

uni_emp = plot(uniform_empiricals..., layout=(length(Ns), length(sparsities)), size=(1200,1200), colorbar=false)
uni_the = plot(uniform_theories..., layout=(length(Ns), length(sparsities)), size=(1200,1200), colorbar=false)
gau_emp = plot(gaussian_empiricals..., layout=(length(Ns), length(sparsities)), size=(1200,1200), colorbar=false)
gau_the = plot(gaussian_theories..., layout=(length(Ns), length(sparsities)), size=(1200,1200), colorbar=false)

plot(uni_emp, uni_the, gau_emp, gau_the, layout=(2,2),size=(2000,1000),
    tickfontsize=34, labelfontsize=34,margin=5mm,colorbar=false)


##################################################
### Speed up for some q values with more trials
q_values = [0.05, 0.15]
results = Dict{Float64, Array{Float64,2}}()
num_networks2 = 100

function calc_sparsity(W)
    N = size(W,1)
    n_nonzero = count(iszero, W)
    return n_nonzero / (N^2)
end

mats = []

for (i,q) in enumerate(q_values)
    println("Running q = $q ...")
    rel_srs = zeros(num_networks2, length(fs))
    for k in 1:num_networks2
        A_base = er_matrix(N, S)
        A = reassign_to_diagonal(copy(A_base), q)
        base_sr = spec_rad(A)

            push!(mats, calc_sparsity(A))

        for (j, frac) in enumerate(fs)
            n_prune = round(Int, frac * N)
            idx_keep = setdiff(1:N, randperm(N)[1:n_prune])
            A_pruned = A[idx_keep, idx_keep]
            rel_srs[k, j] = spec_rad(A_pruned) / base_sr
        end
    end
    results[q] = rel_srs
end

green_colors = [:lightgreen, :green, :darkgreen]

# Plot
plot(size=(700,500))
for (i,q) in enumerate(q_values)
    mean_rel = mean(results[q], dims=1)[:]
    plot!(fs, mean_rel, lw=3, c=green_colors[i], marker=:circle, markersize=5, label="q=$q")
end
plot!(fs, sqrt.(1 .- fs), legend=:bottomleft, grid=false, legendfontsize=14, tickfontsize=16, labelfontsize=16, size=(500,500), title="Spectral Radius Under Random Pruning", xlabel="Fraction pruned", ylabel="Relative spectral radius", lw=3, ls=:dash, c=:black, label="Circular Law Theory")



######################################


using LinearAlgebra, Random, Statistics, Plots

function generate_sparse_matrix(N, S)
    total_nonzeros = Int(round((1 - S) * N^2))
    A = zeros(Float64, N, N)

    idx = randperm(N^2)[1:total_nonzeros]

    for k in idx
        i = div(k-1, N) + 1
        j = mod(k-1, N) + 1
        A[i,j] = rand()*2 - 1
    end

    return A
end


function reassign_to_diagonal(A, q)
    N = size(A,1)

    offdiag = [(i,j) for i in 1:N for j in 1:N if i != j && A[i,j] != 0.0]

    num_to_move = min(round(Int,q*N), length(offdiag))
    num_to_move == 0 && return A

    sel = randperm(length(offdiag))[1:num_to_move]

    zero_diag = [i for i in 1:N if A[i,i] == 0.0]

    for s in sel
        isempty(zero_diag) && break

        i,j = offdiag[s]
        w = A[i,j]
        A[i,j] = 0.0

        d_idx = rand(1:length(zero_diag))
        diag_pos = zero_diag[d_idx]

        A[diag_pos,diag_pos] = w
        deleteat!(zero_diag, d_idx)
    end

    return A
end


function prune_nodes(A, f)
    N = size(A,1)
    n_remove = round(Int, f*N)

    remove = randperm(N)[1:n_remove]
    keep = setdiff(1:N, remove)

    return A[keep, keep]
end


function gershgorin_data(A)

    centers = diag(A)
    radii = vec(sum(abs.(A), dims=2)) .- abs.(centers)

    return centers, radii
end



N=10
S=0.5
q=0.5
prune_levels=[0.0,0.5,0.9]
M_base = generate_sparse_matrix(N,S)

M_zero_diag = copy(M_base)
for i in 1:N
    M_zero_diag[i,i] = 0.0
end

D = reassign_to_diagonal(copy(M_base), q)

plots = []

for f in prune_levels

    M_pruned = prune_nodes(M_zero_diag,f)
    MD_pruned = prune_nodes(D,f)

    matrices = [M_pruned, MD_pruned]

    centers_all = Float64[]
    radii_all = Float64[]

    for mat in matrices
        c,r = gershgorin_data(mat)
        append!(centers_all,c)
        append!(radii_all,r)
    end

    xmin = minimum(centers_all .- radii_all)
    xmax = maximum(centers_all .+ radii_all)

    ymax = maximum(radii_all)
    ymin = -ymax

    padding = 0.1 * max(xmax-xmin, ymax-ymin)

    rowplots = []

    for (col,mat) in enumerate(matrices)

        centers,radii = gershgorin_data(mat)

        p = plot(legend=false, aspect_ratio=1)

        θ = range(0,2π,length=100)

        for (c,r) in zip(centers,radii)

            x = c .+ r*cos.(θ)
            y = r*sin.(θ)

            plot!(p,x,y, color=:blue, alpha=0.2, lw=1)
            scatter!(p,[c],[0], color=:red, markersize=3)
        end

        eigs = eigvals(mat)

        scatter!(p, real.(eigs), imag.(eigs),
                    color=:black, markersize=3)

        idx_max = argmax(abs.(eigs))

        scatter!(p,
            [real(eigs[idx_max])],
            [imag(eigs[idx_max])],
            color=:lime, markersize=6)

        hline!(p,[0],color=:black,lw=1)
        vline!(p,[0],color=:black,lw=1)

        xlims!(p, xmin-padding, xmax+padding)
        ylims!(p, ymin-padding, ymax+padding)

        title = col==1 ? "Fragile (zero diag)" : "Robust (diag reassigned)"
        title!(p,title)

        push!(rowplots,p)
    end

    push!(plots, rowplots...)
end

limm=4
pp = plot(plots[1],plots[3],plots[5],plots[2],plots[4],plots[6], lw=3, tickfontsize=14, layout=(2,3), aspect_ratio=:true, size=(1200,800), grid=false)

