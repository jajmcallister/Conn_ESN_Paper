using Statistics, Base.Threads, JLD2, HypothesisTests
using .ReservoirTasks

# Spectral Radius and Pruning by weighted task variance

function prune_nodes_from_network(weight_matrix, nodes_to_prune)
    orig_size = size(weight_matrix, 1)
    new_size = orig_size - length(nodes_to_prune)
    pruned_matrix = zeros(Float64, new_size, new_size)
    remaining_indices = setdiff(1:orig_size, nodes_to_prune)
    for (new_i, orig_i) in enumerate(remaining_indices)
        for (new_j, orig_j) in enumerate(remaining_indices)
            pruned_matrix[new_i, new_j] = weight_matrix[orig_i, orig_j]
        end
    end
    return pruned_matrix
end

numres=3
conn_srs = Vector{Any}(undef, 39)
er_srs = Vector{Any}(undef, 39)
cfg_srs = Vector{Any}(undef, 39)


@threads for i in 1:39

    task_srs = []

    for taskid in 1:8

        arr = []

        for subneti in 1:numres

            m = deepcopy(conn_ESNs[i][subneti])
            sorted_wtv = sortperm(deepcopy(weighted_tv_conn[i][taskid][subneti]))

            srs = [maximum(abs.(eigvals(prune_nodes_from_network(m, sorted_wtv[1:j])))) for j in 1:(size(m,1)-1)]

            push!(arr, srs)

        end
        push!(task_srs, arr)
    end
    conn_srs[i] = task_srs
end
@threads for i in 1:39

    task_srs = []

    for taskid in 1:8

        arr = []

        for subneti in 1:numres

            m = deepcopy(er_ESNs[i][subneti])
            sorted_wtv = sortperm(deepcopy(weighted_tv_rand[i][taskid][subneti]))

            srs = [maximum(abs.(eigvals(prune_nodes_from_network(m, sorted_wtv[1:j])))) for j in 1:(size(m,1)-1)]

            push!(arr, srs)

        end
        push!(task_srs, arr)
    end
    er_srs[i] = task_srs
end
@threads for i in 1:39

    task_srs = []

    for taskid in 1:8

        arr = []

        for subneti in 1:numres

            m = deepcopy(cfg_ESNs[i][subneti])
            sorted_wtv = sortperm(deepcopy(weighted_tv_cfg[i][taskid][subneti]))

            srs = [maximum(abs.(eigvals(prune_nodes_from_network(m, sorted_wtv[1:j])))) for j in 1:(size(m,1)-1)]

            push!(arr, srs)

        end
        push!(task_srs, arr)
    end
    cfg_srs[i] = task_srs
end

task_plots = []
for  taskid in 1:8

    p = plot()
    for i in 12:39
        plot!(1/length(mean(er_srs[i][taskid])):1/length(mean(er_srs[i][taskid])):1, mean(er_srs[i][taskid]), c=crimsons[3], alpha=0.8, lw=2, label=false)
    end
    for i in 3:11
        plot!(1/length(mean(er_srs[i][taskid])):1/length(mean(er_srs[i][taskid])):1, mean(er_srs[i][taskid]), c=:crimson, alpha=0.8, lw=2, label="ER")
    end

    for i in 12:39
        plot!(1/length(mean(conn_srs[i][taskid])):1/length(mean(conn_srs[i][taskid])):1, mean(conn_srs[i][taskid]), c=dodgerblues[3], alpha=0.8, lw=2, label=false)
    end
    for i in 3:11
        plot!(1/length(mean(conn_srs[i][taskid])):1/length(mean(conn_srs[i][taskid])):1, mean(conn_srs[i][taskid]), c=:dodgerblue4, alpha=0.8, lw=2, label="Conn")
    end
    plot!(title="Task $taskid", xlabel="Proportion of nodes pruned", ylabel="Spectral Radius", ylim=(0,1.1), xlim=(0,1), yticks=0:0.2:1.2, ytickfontsize=12, xtickfontsize=12, titlefontsize=14, legend=false, size=(400,300), grid=false)

    push!(task_plots, p)
end

plot(task_plots..., layout=(4,2), size=(800,1200), leftmargin=5mm, bottommargin=5mm)


using Interpolations, Statistics, Plots

# define common pruning fraction grid
x_common = range(0, 1, length=100)

# helper to get interpolated mean curve for one group
function mean_spectral_radius(group_srs)
    # group_srs: vector of networks, each = [task][subnet][fraction_index]
    curves = []

    for i in group_srs
        for taskid in 1:8
            # average across subnetworks
            task_mean = mean(i[taskid])
            x = range(0, 1, length=length(task_mean))
            itp = interpolate((x,), task_mean, Gridded(Linear()))
            push!(curves, itp.(x_common))
        end
    end

    return mean(reduce(hcat, curves), dims=2)[:]
end

#Compute for each group 
mean_conn_small = mean_spectral_radius(conn_srs[3:11])
mean_conn_large = mean_spectral_radius(conn_srs[12:39])
sem_conn_small = std(mean_conn_small) / sqrt(length(conn_srs[3:11]))
sem_conn_large = std(mean_conn_large) / sqrt(length(conn_srs[12:39]))

mean_er_small = mean_spectral_radius(er_srs[3:11])
mean_er_large = mean_spectral_radius(er_srs[12:39])
sem_er_small = std(mean_er_small) / sqrt(length(er_srs[3:11]))
sem_er_large = std(mean_er_large) / sqrt(length(er_srs[12:39]))
# mean_er_sr = mean_spectral_radius(er_srs[3:end])

mean_cfg_small = mean_spectral_radius(cfg_srs[3:11])
mean_cfg_large = mean_spectral_radius(cfg_srs[12:39])
sem_cfg_small = std(mean_cfg_small) / sqrt(length(cfg_srs[3:11]))
sem_cfg_large = std(mean_cfg_large) / sqrt(length(cfg_srs[12:39]))
# mean_cfg_srs = mean_spectral_radius(cfg_srs[3:end])


plot(x_common, mean_er_sr, c=:crimson, lw=5, label=false)
plot!(x_common, mean_cfg_srs, c=:orange, lw=5, label=false)
plot!(x_common, mean_conn_small, c=:dodgerblue4, lw=5, label=false)
plot!(x_common, mean_conn_large, c=dodgerblues[3], lw=5, label=false)
plot!(xlabel="Proportion of nodes pruned", ylabel="Spectral radius",
      ylim=(0,1.1), xlim=(0,1), legend=true, grid=false, size=(600,400), title="Mean Spectral Radius of Networks while Pruning Nodes by WTV",)


p1 = plot(x_common, mean_er_small, c=:crimson, lw=5, label=false)
plot!(x_common, mean_cfg_small, c=:orange, lw=5, label=false)
plot!(x_common, mean_conn_small, c=:dodgerblue4, lw=5, label=false)
plot!(xlabel="Proportion of nodes pruned",
      ylim=(0,1.1), xlim=(0,1), legend=true, grid=false, size=(600,400), title="Larva",)
p2 = plot(x_common, mean_er_large, c=crimsons[3], lw=5, label=false)
plot!(x_common, mean_cfg_large, c=oranges[3], lw=5, label=false)
plot!(x_common, mean_conn_large, c=dodgerblues[3], lw=5, label=false)
plot!(xlabel="Proportion of nodes pruned",
      ylim=(0,1.1), xlim=(0,1), legend=true, grid=false, size=(600,400), title="Adult",)

      pp = plot(p1, p2, layout=(1,2), size=(700,400), leftmargin=5mm, bottommargin=5mm, labelfontsize=12, titlefontsize=20, tickfontsize=14)


function theory_sr_prune_rand(N,m_vals)
    v = []
    for m in m_vals
        push!(v, sqrt(1 - m/N))
    end
    return v
end

mvals = 0:1:99
theory_sr = theory_sr_prune_rand(100, mvals)


p1 = plot(0.01:1/100:1, theory_sr, c=:black, lw=3, label="Theoretical Approximation under Random Pruning", linestyle=:dash)
plot!(x_common, mean_er_small, c=:crimson,lw=5, label=false)
plot!(x_common, mean_cfg_small,  c=:orange, lw=5, label=false)
plot!(x_common, mean_conn_small, c=:dodgerblue4, lw=5, label=false)
plot!(x_common, mean_conn_small, c=:dodgerblue4, lw=5, label="Conn", linestyle=:solid)
plot!(x_common, mean_er_small, c=:crimson, lw=5, label="ER", linestyle=:solid)
plot!(x_common, mean_cfg_small, c=:orange, lw=5, label="CFG", linestyle=:solid)
plot!(0.01:1/100:1, theory_sr, c=:black, lw=3, label=false, linestyle=:dash)
plot!(xlabel="Proportion of nodes pruned",
      ylim=(0,1.1), xlim=(0,1), legend=true, grid=false, size=(600,400), title="Larva",)


p2 = plot(0.01:1/100:1, theory_sr, c=:black, lw=3, label="Theoretical Approximation under Random Pruning", linestyle=:dash)
plot!(x_common, mean_er_large, c=crimsons[3], lw=5, label=false)
plot!(x_common, mean_cfg_large, c=oranges[3], lw=5  ,label=false)
plot!(x_common, mean_conn_large, c=dodgerblues[3], lw=5, label=false)
plot!(x_common, mean_conn_large, c=dodgerblues[3], lw=5, label="Conn", linestyle=:solid)
plot!(x_common, mean_er_large, c=crimsons[3], lw=5, label="ER", linestyle=:solid)
plot!(x_common, mean_cfg_large, c=oranges[3], lw=5, label="CFG", linestyle=:solid)
plot!(0.01:1/100:1, theory_sr, c=:black, lw=3, label=false, linestyle=:dash)
plot!(xlabel="Proportion of nodes pruned",
      ylim=(0,1.1), xlim=(0,1), legend=true, grid=false, size=(600,400), title="Adult",)


pp = plot(p1, p2, layout=(1,2), size=(700,400), legend=false, leftmargin=5mm, bottommargin=5mm, labelfontsize=12, titlefontsize=20, tickfontsize=14)


h1 = heatmap(vv_conns[11][3]',xticks=false, cbar=false,title="Conn", yflip=true, xlabel="Neurons", yticks=1:8,  ylabel="Task ID",size=(600,600),c=:viridis)
h2 = heatmap(vv_rands[11][3]', xticks=false,cbar=false, title="ER", yflip=true, xlabel="Neurons", yticks=1:8, c=:viridis)

plot(h1,h2, layout=(1,2),size=(500,200), leftmargin=5mm, bottommargin=5mm)


###################################
## Performance and Pruning by weighted task variance



function prune_nodes_from_network(weight_matrix, nodes_to_prune)
    orig_size = size(weight_matrix, 1)
    keep_mask = ones(Bool, orig_size)
    keep_mask[nodes_to_prune] .= false
    return weight_matrix[keep_mask, keep_mask]
end

bottom_id = 3
top_id = 39
numres = 30
perf_drop_frac = 0.1      
tol_frac = 1e-2         

# allocation of arrays
results_fractions = Dict()
results_history = Dict()

tasks = [:memory, :recall, :dm, :ddm, :osc, :lot, :lor, :ros]
types = [:conn, :er, :cfg]

for t in tasks
    for ty in types
        key = Symbol("$(ty)_$(t)")
        results_fractions[key] = Vector{Any}(undef, 39)
        results_history[key] = Vector{Any}(undef, 39)
    end
end


# Working Memory capacity
conn_overall_memory_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
er_overall_memory_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
conn_memory_fractions = Vector{Float64}(undef, 39)
er_memory_fractions   = Vector{Float64}(undef, 39)
ESN_collection_mem = [
    (conn_ESNs, input_conn, reg_conn, leak_conn, weighted_tv_conn, conn_memory_fractions, conn_overall_memory_pruning_performances),
    (er_ESNs,   input_er,   reg_er,   leak_er,   weighted_tv_rand, er_memory_fractions, er_overall_memory_pruning_performances)
]
@threads for subnetid in bottom_id:top_id
    for (ESN_set, in_set, reg_set, leak_set, tv_set, frac_vec, perf_dict) in ESN_collection_mem
        starting_perf, _ = ReservoirTasks.res_performance_memory(ESN_set[subnetid][1:numres], 2, in_set[subnetid,1], reg_set[subnetid,1], leak_set[subnetid,1])
        baseline = mean(starting_perf)
        thresh = (1 - perf_drop_frac) * baseline
        num_neurons = size(ESN_set[subnetid][1], 1)
        sorted_neurons = [sortperm(tv_set[subnetid][1][real_id]) for real_id in 1:numres]

        low_frac, high_frac, best_frac = 0.0, 1.0, NaN
        history = Dict{Float64, Vector{Float64}}()

        while (high_frac - low_frac) > tol_frac
            mid_frac = (low_frac + high_frac)/2
            num_prune = max(1, round(Int, mid_frac * num_neurons))
            cropped = [prune_nodes_from_network(ESN_set[subnetid][r], sorted_neurons[r][1:num_prune]) for r in 1:numres]
            perf, _ = ReservoirTasks.res_performance_memory(cropped, 2, in_set[subnetid,1], reg_set[subnetid,1], leak_set[subnetid,1])
            history[mid_frac] = perf
            if mean(perf) < thresh; best_frac = mid_frac; high_frac = mid_frac; else; low_frac = mid_frac; end
        end
        frac_vec[subnetid], perf_dict[subnetid] = (isnan(best_frac) ? 1.0 : best_frac), history
    end
end

p1 = plot(conn_memory_fractions[3:end], c=:blue)
plot!(er_memory_fractions[3:end], c=:red)

# can do SignedRankTest(conn_memory_fractions[3:11], er_memory_fractions[3:11]) etc 
#to get p-values for differences in pruning fractions 



#Sequence Recall 
conn_overall_recall_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
er_overall_recall_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
conn_recall_fractions = Vector{Float64}(undef, 39)
er_recall_fractions   = Vector{Float64}(undef, 39)
ESN_collection_rec = [
    (conn_ESNs, input_conn, reg_conn, leak_conn, weighted_tv_conn, conn_recall_fractions, conn_overall_recall_pruning_performances),
    (er_ESNs,   input_er,   reg_er,   leak_er,   weighted_tv_rand, er_recall_fractions,   er_overall_recall_pruning_performances)
]
Ls = collect(10:5:100); pushfirst!(Ls, 1:9...)
@threads for subnetid in bottom_id:top_id
    for (ESN_set, in_set, reg_set, leak_set, tv_set, frac_vec, perf_dict) in ESN_collection_rec
        starting_perf, _ = ReservoirTasks.res_performance_recall(ESN_set[subnetid][1:numres], 2, in_set[subnetid,2], reg_set[subnetid,2], leak_set[subnetid,2], Ls; threshold=0.6)
        starting_perf = vcat(starting_perf...)
        thresh = (1 - perf_drop_frac) * mean(starting_perf)
        num_neurons = size(ESN_set[subnetid][1], 1)
        sorted_neurons = [sortperm(tv_set[subnetid][2][r]) for r in 1:numres]

        low_frac, high_frac, best_frac = 0.0, 1.0, NaN
        history = Dict{Float64, Vector{Any}}()

        while (high_frac - low_frac) > tol_frac
            mid_frac = (low_frac + high_frac)/2
            num_prune = max(1, round(Int, mid_frac * num_neurons))
            cropped = [prune_nodes_from_network(ESN_set[subnetid][r], sorted_neurons[r][1:num_prune]) for r in 1:numres]
            perf, _ = ReservoirTasks.res_performance_recall(cropped, 2, in_set[subnetid,2], reg_set[subnetid,2], leak_set[subnetid,2], Ls; threshold=0.6)
            perf = vcat(perf...)
            history[mid_frac] = perf
            if mean(perf) < thresh; best_frac = mid_frac; high_frac = mid_frac; else; low_frac = mid_frac; end
        end
        frac_vec[subnetid], perf_dict[subnetid] = (isnan(best_frac) ? 1.0 : best_frac), history
    end
end

p2 = plot(conn_recall_fractions[3:end], c=:blue)
plot!(er_recall_fractions[3:end], c=:red)


#Decision Making
conn_overall_dm_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
er_overall_dm_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
conn_dm_fractions = Vector{Float64}(undef, 39)
er_dm_fractions   = Vector{Float64}(undef, 39)
ESN_collection_dm = [
    (conn_ESNs, input_conn, reg_conn, leak_conn, weighted_tv_conn, conn_dm_fractions, conn_overall_dm_pruning_performances),
    (er_ESNs,   input_er,   reg_er,   leak_er,   weighted_tv_rand, er_dm_fractions,   er_overall_dm_pruning_performances)
    ]
biases = vcat(reverse(0.55:0.05:1.0), reverse(0.1:0.02:0.5), reverse(0.0:0.005:0.99))
@threads for subnetid in bottom_id:top_id
    for (ESN_set, in_set, reg_set, leak_set, tv_set, frac_vec, perf_dict) in ESN_collection_dm
        starting_perf = ReservoirTasks.res_performance_decisionmaking(ESN_set[subnetid][1:numres], 2, in_set[subnetid,3], reg_set[subnetid,3], leak_set[subnetid,3], biases; threshold=0.8)
        starting_perf = vcat(starting_perf...)
        thresh = (1 - perf_drop_frac) * mean(starting_perf)
        num_neurons = size(ESN_set[subnetid][1], 1)
        sorted_neurons = [sortperm(tv_set[subnetid][3][r]) for r in 1:numres]

        low_frac, high_frac, best_frac = 0.0, 1.0, NaN
        history = Dict{Float64, Vector{Any}}()

        while (high_frac - low_frac) > tol_frac
            mid_frac = (low_frac + high_frac)/2
            num_prune = max(1, round(Int, mid_frac * num_neurons))
            cropped = [prune_nodes_from_network(ESN_set[subnetid][r], sorted_neurons[r][1:num_prune]) for r in 1:numres]
            perf = ReservoirTasks.res_performance_decisionmaking(cropped, 2, in_set[subnetid,3], reg_set[subnetid,3], leak_set[subnetid,3], biases; threshold=0.8)
            perf = vcat(perf...)
            history[mid_frac] = perf
            if mean(perf) < thresh; best_frac = mid_frac; high_frac = mid_frac; else; low_frac = mid_frac; end
        end
        frac_vec[subnetid], perf_dict[subnetid] = (isnan(best_frac) ? 1.0 : best_frac), history
    end
end

p3 = plot(conn_dm_fractions[3:end], c=:blue)
plot!(er_dm_fractions[3:end], c=:red)


#Delayed Decision Making
conn_overall_ddm_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
er_overall_ddm_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
conn_ddm_fractions = Vector{Float64}(undef, 39)
er_ddm_fractions   = Vector{Float64}(undef, 39)
ESN_collection_ddm = [
    (conn_ESNs, input_conn, reg_conn, leak_conn, weighted_tv_conn, conn_ddm_fractions, conn_overall_ddm_pruning_performances),
    (er_ESNs,   input_er,   reg_er,   leak_er,   weighted_tv_rand, er_ddm_fractions,   er_overall_ddm_pruning_performances)
    ]
variances = vcat(0.01:0.01,0.09, 0.1:0.2:1., 1.1:0.1:1.5, 1.55:0.05:4)
@threads for subnetid in bottom_id:top_id
    for (ESN_set, in_set, reg_set, leak_set, tv_set, frac_vec, perf_dict) in ESN_collection_ddm
        starting_perf = ReservoirTasks.res_performance_delay_decisionmaking(ESN_set[subnetid][1:numres], 2, in_set[subnetid,4], reg_set[subnetid,4], leak_set[subnetid,4], variances; threshold=0.8)
        starting_perf = vcat(starting_perf...)
        thresh = (1 - perf_drop_frac) * mean(starting_perf)
        num_neurons = size(ESN_set[subnetid][1], 1)
        sorted_neurons = [sortperm(tv_set[subnetid][4][r]) for r in 1:numres]

        low_frac, high_frac, best_frac = 0.0, 1.0, NaN
        history = Dict{Float64, Vector{Float64}}()

        while (high_frac - low_frac) > tol_frac
            mid_frac = (low_frac + high_frac)/2
            num_prune = max(1, round(Int, mid_frac * num_neurons))
            cropped = [prune_nodes_from_network(ESN_set[subnetid][r], sorted_neurons[r][1:num_prune]) for r in 1:numres]
            perf = ReservoirTasks.res_performance_delay_decisionmaking(cropped, 2, in_set[subnetid,4], reg_set[subnetid,4], leak_set[subnetid,4], variances; threshold=0.8)
            perf = vcat(perf...)
            history[mid_frac] = perf
            if mean(perf) < thresh; best_frac = mid_frac; high_frac = mid_frac; else; low_frac = mid_frac; end
        end
        frac_vec[subnetid], perf_dict[subnetid] = (isnan(best_frac) ? 1.0 : best_frac), history
    end
end

p4 = plot(conn_ddm_fractions[3:end], c=:blue)
plot!(er_ddm_fractions[3:end], c=:red)

#Oscillator
conn_overall_osc_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
er_overall_osc_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
conn_osc_fractions = Vector{Float64}(undef, 39)
er_osc_fractions   = Vector{Float64}(undef, 39)
ESN_collection_osc = [
    (conn_ESNs, input_conn, reg_conn, leak_conn, weighted_tv_conn, conn_osc_fractions, conn_overall_osc_pruning_performances),
    (er_ESNs,   input_er,   reg_er,   leak_er,   weighted_tv_rand, er_osc_fractions,   er_overall_osc_pruning_performances)
    ]
@threads for subnetid in bottom_id:top_id
    for (ESN_set, in_set, reg_set, leak_set, tv_set, frac_vec, perf_dict) in ESN_collection_osc
        starting_perf, _ = ReservoirTasks.res_performance_oscillator(ESN_set[subnetid][1:numres], osc_data, 0.5, 2, in_set[subnetid,5], reg_set[subnetid,5], leak_set[subnetid,5])
        thresh = (1 - perf_drop_frac) * mean(starting_perf)
        num_neurons = size(ESN_set[subnetid][1], 1)
        sorted_neurons = [sortperm(tv_set[subnetid][5][r]) for r in 1:numres]

        low_frac, high_frac, best_frac = 0.0, 1.0, NaN
        history = Dict{Float64, Vector{Float64}}()

        while (high_frac - low_frac) > tol_frac
            mid_frac = (low_frac + high_frac)/2
            num_prune = max(1, round(Int, mid_frac * num_neurons))
            cropped = [prune_nodes_from_network(ESN_set[subnetid][r], sorted_neurons[r][1:num_prune]) for r in 1:numres]
            perf, _ = ReservoirTasks.res_performance_oscillator(cropped, osc_data, 0.5, 2, in_set[subnetid,5], reg_set[subnetid,5], leak_set[subnetid,5])
            history[mid_frac] = perf
            if mean(perf) < thresh; best_frac = mid_frac; high_frac = mid_frac; else; low_frac = mid_frac; end
        end
        frac_vec[subnetid], perf_dict[subnetid] = (isnan(best_frac) ? 1.0 : best_frac), history
    end
end

p5 = plot(conn_osc_fractions[3:end], c=:blue)
plot!(er_osc_fractions[3:end], c=:red)

#Lotka-Volterra

conn_overall_lot_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
er_overall_lot_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
conn_lot_fractions = Vector{Float64}(undef, 39)
er_lot_fractions   = Vector{Float64}(undef, 39)
ESN_collection_lot = [
    (conn_ESNs, input_conn, reg_conn, leak_conn, weighted_tv_conn, conn_lot_fractions, conn_overall_lot_pruning_performances),
    (er_ESNs,   input_er,   reg_er,   leak_er,   weighted_tv_rand, er_lot_fractions,   er_overall_lot_pruning_performances)
    ]
@threads for subnetid in bottom_id:top_id
    for (ESN_set, in_set, reg_set, leak_set, tv_set, frac_vec, perf_dict) in ESN_collection_lot
        starting_perf, _ = ReservoirTasks.res_performance_lotka_2d(ESN_set[subnetid][1:numres], lotka_data, 0.5, 2, in_set[subnetid,6], reg_set[subnetid,6], leak_set[subnetid,6])
        thresh = (1 - perf_drop_frac) * mean(starting_perf)
        num_neurons = size(ESN_set[subnetid][1], 1)
        sorted_neurons = [sortperm(tv_set[subnetid][6][r]) for r in 1:numres]
        low_frac, high_frac, best_frac = 0.0, 1.0, NaN
        history = Dict{Float64, Vector{Float64}}()

        while (high_frac - low_frac) > tol_frac
            mid_frac = (low_frac + high_frac)/2
            num_prune = max(1, round(Int, mid_frac * num_neurons))
            cropped = [prune_nodes_from_network(ESN_set[subnetid][r], sorted_neurons[r][1:num_prune]) for r in 1:numres]
            perf, _ = ReservoirTasks.res_performance_lotka_2d(cropped, lotka_data, 0.5, 2, in_set[subnetid,6], reg_set[subnetid,6], leak_set[subnetid,6])
            history[mid_frac] = perf
            if mean(perf) < thresh; best_frac = mid_frac; high_frac = mid_frac; else; low_frac = mid_frac; end
        end
        frac_vec[subnetid], perf_dict[subnetid] = (isnan(best_frac) ? 1.0 : best_frac), history
    end
end

p6 = plot(conn_lot_fractions[3:end], c=:blue)
plot!(er_lot_fractions[3:end], c=:red)



#Lorenz
conn_overall_lor_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
er_overall_lor_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
conn_lor_fractions = Vector{Float64}(undef, 39)
er_lor_fractions   = Vector{Float64}(undef, 39)
ESN_collection_lor = [
    (conn_ESNs, input_conn, reg_conn, leak_conn, weighted_tv_conn, conn_lor_fractions, conn_overall_lor_pruning_performances),
    (er_ESNs,   input_er,   reg_er,   leak_er,   weighted_tv_rand, er_lor_fractions,   er_overall_lor_pruning_performances)
    ]
@threads for subnetid in bottom_id:top_id
    for (ESN_set, in_set, reg_set, leak_set, tv_set, frac_vec, perf_dict) in ESN_collection_lor
        starting_perf, _ = ReservoirTasks.res_performance_lorenz(ESN_set[subnetid][1:numres], lorenz_train_data, lorenz_test_data, 2, 0.5, in_set[subnetid,7], reg_set[subnetid,7], leak_set[subnetid,7])
        thresh = (1 - perf_drop_frac) * mean(starting_perf)
        num_neurons = size(ESN_set[subnetid][1], 1)
        sorted_neurons = [sortperm(tv_set[subnetid][7][r]) for r in 1:numres]

        low_frac, high_frac, best_frac = 0.0, 1.0, NaN
        history = Dict{Float64, Vector{Float64}}()

        while (high_frac - low_frac) > tol_frac
            mid_frac = (low_frac + high_frac)/2
            num_prune = max(1, round(Int, mid_frac * num_neurons))
            cropped = [prune_nodes_from_network(ESN_set[subnetid][r], sorted_neurons[r][1:num_prune]) for r in 1:numres]
            perf, _ = ReservoirTasks.res_performance_lorenz(cropped, lorenz_train_data, lorenz_test_data, 2, 0.5, in_set[subnetid,7], reg_set[subnetid,7], leak_set[subnetid,7])
            history[mid_frac] = perf
            if mean(perf) < thresh; best_frac = mid_frac; high_frac = mid_frac; else; low_frac = mid_frac; end
        end
        frac_vec[subnetid], perf_dict[subnetid] = (isnan(best_frac) ? 1.0 : best_frac), history
    end
end

p7 = plot(conn_lor_fractions[3:end], c=:blue)
plot!(er_lor_fractions[3:end], c=:red)

#Rossler
conn_overall_ros_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
er_overall_ros_pruning_performances =
    Vector{Dict{Float64, Vector{Float64}}}(undef, 39)
conn_ros_fractions = Vector{Float64}(undef, 39)
er_ros_fractions   = Vector{Float64}(undef, 39)
ESN_collection_ros = [
    (conn_ESNs, input_conn, reg_conn, leak_conn, weighted_tv_conn, conn_ros_fractions, conn_overall_ros_pruning_performances),
    (er_ESNs,   input_er,   reg_er,   leak_er,   weighted_tv_rand, er_ros_fractions,   er_overall_ros_pruning_performances)
    ]
@threads for subnetid in bottom_id:top_id
    for (ESN_set, in_set, reg_set, leak_set, tv_set, frac_vec, perf_dict) in ESN_collection_ros
        starting_perf, _ = ReservoirTasks.res_performance_rossler(ESN_set[subnetid][1:numres], rossler_train_data, rossler_test_data, 2, 0.5, in_set[subnetid,8], reg_set[subnetid,8], leak_set[subnetid,8])
        thresh = (1 - perf_drop_frac) * mean(starting_perf)
        num_neurons = size(ESN_set[subnetid][1], 1)
        sorted_neurons = [sortperm(tv_set[subnetid][8][r]) for r in 1:numres]

        low_frac, high_frac, best_frac = 0.0, 1.0, NaN
        history = Dict{Float64, Vector{Float64}}()

        while (high_frac - low_frac) > tol_frac
            mid_frac = (low_frac + high_frac)/2
            num_prune = max(1, round(Int, mid_frac * num_neurons))
            cropped = [prune_nodes_from_network(ESN_set[subnetid][r], sorted_neurons[r][1:num_prune]) for r in 1:numres]
            perf, _ = ReservoirTasks.res_performance_rossler(cropped, rossler_train_data, rossler_test_data, 2, 0.5, in_set[subnetid,8], reg_set[subnetid,8], leak_set[subnetid,8])
            history[mid_frac] = perf
            if mean(perf) < thresh; best_frac = mid_frac; high_frac = mid_frac; else; low_frac = mid_frac; end
        end
        frac_vec[subnetid], perf_dict[subnetid] = (isnan(best_frac) ? 1.0 : best_frac), history
    end
end

p8 = plot(conn_ros_fractions[3:end], c=:blue)
plot!(er_ros_fractions[3:end], c=:red)


############


p = plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(4,2))




b1 = bar([1], [mean(conn_memory_fractions[3:11])], yerr=[std(conn_memory_fractions[3:11]) / sqrt(9)], 
        color=:dodgerblue4, alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Memory Task")
bar!([1.5], [mean(conn_memory_fractions[12:end])], yerr=[std(conn_memory_fractions[12:end]) / sqrt(28)], 
        color=dodgerblues[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Memory Task")
bar!([3], [mean(er_memory_fractions[3:11])], yerr=[std(er_memory_fractions[3:11]) / sqrt(9)],
        color=crimsons[2], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Memory Task")
bar!([3.5], [mean(er_memory_fractions[12:end])], yerr=[std(er_memory_fractions[12:end]) / sqrt(28)],
        color=crimsons[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Memory Task")
        plot!(ylim=(0.,0.6))
b2 = bar([1], [mean(conn_recall_fractions[3:11])], yerr=[std(conn_recall_fractions[3:11]) / sqrt(9)],
        color=:dodgerblue4, alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Recall Task")
bar!([1.5], [mean(conn_recall_fractions[12:end])], yerr=[std(conn_recall_fractions[12:end]) / sqrt(28)],
        color=dodgerblues[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Recall Task")
bar!([3], [mean(er_recall_fractions[3:11])], yerr=[std(er_recall_fractions[3:11]) / sqrt(9)],
        color=crimsons[2], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Recall Task")
bar!([3.5], [mean(er_recall_fractions[12:end])], yerr=[std(er_recall_fractions[12:end]) / sqrt(28)],
        color=crimsons[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Recall Task")
        plot!(ylim=(0,0.45))
b3 = bar([1], [mean(conn_dm_fractions[3:11])], yerr=[std(conn_dm_fractions[3:11]) / sqrt(9)],
        color=:dodgerblue4, alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Decision Making Task")
bar!([1.5], [mean(conn_dm_fractions[12:end])], yerr=[std(conn_dm_fractions[12:end]) / sqrt(28)],
        color=dodgerblues[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Decision Making Task")
bar!([3], [mean(er_dm_fractions[3:11])], yerr=[std(er_dm_fractions[3:11]) / sqrt(9)],
        color=crimsons[2], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Decision Making Task")
bar!([3.5], [mean(er_dm_fractions[12:end])], yerr=[std(er_dm_fractions[12:end]) / sqrt(28)],
        color=crimsons[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Decision Making Task")
        plot!(ylim=(0.6,1.05))
b4 = bar([1], [mean(conn_ddm_fractions[3:11])], yerr=[std(conn_ddm_fractions[3:11]) / sqrt(9)],
        color=:dodgerblue4, alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Delayed Decision Making Task")
bar!([1.5], [mean(conn_ddm_fractions[12:end])], yerr=[std(conn_ddm_fractions[12:end]) / sqrt(28)],
        color=dodgerblues[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Delayed Decision Making Task")
bar!([3], [mean(er_ddm_fractions[3:11])], yerr=[std(er_ddm_fractions[3:11]) / sqrt(9)],
        color=crimsons[2], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Delayed Decision Making Task")
bar!([3.5], [mean(er_ddm_fractions[12:end])], yerr=[std(er_ddm_fractions[12:end]) / sqrt(28)],
        color=crimsons[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Delayed Decision Making Task")
plot!(ylim=(0.2,0.8))
b5 = bar([1], [mean(conn_osc_fractions[3:11])], yerr=[std(conn_osc_fractions[3:11]) / sqrt(9)],
        color=:dodgerblue4, alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Oscillator Task")
bar!([1.5], [mean(conn_osc_fractions[12:end])], yerr=[std(conn_osc_fractions[12:end]) / sqrt(28)],
        color=dodgerblues[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Oscillator Task")
bar!([3], [mean(er_osc_fractions[3:11])], yerr=[std(er_osc_fractions[3:11]) / sqrt(9)],
        color=crimsons[2], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Oscillator Task")  
bar!([3.5], [mean(er_osc_fractions[12:end])], yerr=[std(er_osc_fractions[12:end]) / sqrt(28)],
        color=crimsons[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Oscillator Task")
plot!(ylim=(0.2,0.67))
        b6 = bar([1], [mean(conn_lot_fractions[3:11])], yerr=[std(conn_lot_fractions[3:11]) / sqrt(9)],
        color=:dodgerblue4, alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Lotka-Volterra Task")
bar!([1.5], [mean(conn_lot_fractions[12:end])], yerr=[std(conn_lot_fractions[12:end]) / sqrt(28)],
        color=dodgerblues[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Lotka-Volterra Task")
bar!([3], [mean(er_lot_fractions[3:11])], yerr=[std(er_lot_fractions[3:11]) / sqrt(9)],
        color=crimsons[2], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Lotka-Volterra Task")
bar!([3.5], [mean(er_lot_fractions[12:end])], yerr=[std(er_lot_fractions[12:end]) / sqrt(28)],
        color=crimsons[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Lotka-Volterra Task")
plot!(ylim=(0.2,0.8))
        b7 = bar([1], [mean(conn_lor_fractions[3:11])], yerr=[std(conn_lor_fractions[3:11]) / sqrt(9)],
        color=:dodgerblue4, alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Lorenz Task")
bar!([1.5], [mean(conn_lor_fractions[12:end])], yerr=[std(conn_lor_fractions[12:end]) / sqrt(28)],
        color=dodgerblues[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Lorenz Task")   
bar!([3], [mean(er_lor_fractions[3:11])], yerr=[std(er_lor_fractions[3:11]) / sqrt(9)],
        color=crimsons[2], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Lorenz Task")
bar!([3.5], [mean(er_lor_fractions[12:end])], yerr=[std(er_lor_fractions[12:end]) / sqrt(28)],
        color=crimsons[3], alpha=0.8, legend=false, ylabel="Fraction Pruned", title="Lorenz Task")
plot!(ylim=(0.,0.18))
        b8 = bar([1], [mean(conn_ros_fractions[3:11])], yerr=[std(conn_ros_fractions[3:11]) / sqrt(9)],
        color=:dodgerblue4, alpha=0.8, legend=false,  ylabel="Fraction Pruned", title="Rossler Task")
bar!([1.5], [mean(conn_ros_fractions[12:end])], yerr=[std(conn_ros_fractions[12:end]) / sqrt(28)],
        color=dodgerblues[3], alpha=0.8, legend=false,  ylabel="Fraction Pruned", title="Rossler Task") 
bar!([3], [mean(er_ros_fractions[3:11])], yerr=[std(er_ros_fractions[3:11]) / sqrt(9)],
        color=crimsons[2], alpha=0.8, legend=false,  ylabel="Fraction Pruned", title="Rossler Task")
bar!([3.5], [mean(er_ros_fractions[12:end])], yerr=[std(er_ros_fractions[12:end]) / sqrt(28)],
        color=crimsons[3], alpha=0.8, legend=false,  ylabel="Fraction Pruned", title="Rossler Task")
plot!(ylim=(0.2,0.8))

        p  = plot(b1, b2, b3, b4, b5, b6, b7, b8, lw=3, layout=(4,2),size=(1000,1200), grid=false)