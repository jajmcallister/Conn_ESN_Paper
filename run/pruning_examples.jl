function prune_nodes_from_network(weight_matrix, nodes_to_prune)
    orig_size = size(weight_matrix, 1)
    keep_mask = ones(Bool, orig_size)
    keep_mask[nodes_to_prune] .= false
    return weight_matrix[keep_mask, keep_mask]
end

function moving_average(x, k)
    n = length(x)
    y = similar(x, n - k + 1)
    for i in 1:length(y)
        y[i] = mean(@view x[i:i+k-1])
    end
    return y
end



using Main.Threads
using ProgressLogging
using Statistics


# MEMORY

subnetid = 6
num_res = 30

starting_perf_conn, _ = ReservoirTasks.res_performance_memory(conn_ESNs[subnetid][1:numres], 2, input_conn[subnetid,1], reg_conn[subnetid,1], leak_conn[subnetid,1])
baseline = mean(starting_perf_conn)
ordered_neurons_conn = [sortperm(weighted_tv_conn[subnetid][1][id]) for id in 1:num_res] 
conn_mem_eg = Vector{Any}(undef, length(1:2:size(conn_ESNs[subnetid][1], 1)))
@threads for i in 1:2:size(conn_ESNs[subnetid][1], 1)
    cropped = [prune_nodes_from_network(conn_ESNs[subnetid][r], ordered_neurons_conn[r][1:i]) for r in 1:num_res]
    perf, _ = ReservoirTasks.res_performance_memory(cropped, 2, input_conn[subnetid,1], reg_conn[subnetid,1], leak_conn[subnetid,1])
    conn_mem_eg[i ÷ 2 + 1] = mean(perf)/baseline
end
            
starting_perf_er, _ = ReservoirTasks.res_performance_memory(er_ESNs[subnetid][1:numres], 2, input_er[subnetid,1], reg_er[subnetid,1], leak_er[subnetid,1])
baseline = mean(starting_perf_er)
ordered_neurons_er = [sortperm(weighted_tv_rand[subnetid][1][id]) for id in 1:num_res]
er_mem_eg = Vector{Any}(undef, length(1:2:size(er_ESNs[subnetid][1], 1)))
@threads for i in 1:2:size(er_ESNs[subnetid][1], 1)
    cropped = [prune_nodes_from_network(er_ESNs[subnetid][r], ordered_neurons_er[r][1:i]) for r in 1:num_res]
    perf, _ = ReservoirTasks.res_performance_memory(cropped, 2, input_er[subnetid,1], reg_er[subnetid,1], leak_er[subnetid,1])
    er_mem_eg[i ÷ 2 + 1] = mean(perf)/baseline
end

adult_subnetid = 30
starting_perf_conn_adult, _ = ReservoirTasks.res_performance_memory(conn_ESNs[adult_subnetid][1:numres], 2, input_conn[adult_subnetid,1], reg_conn[adult_subnetid,1], leak_conn[adult_subnetid,1])
baseline = mean(starting_perf_conn_adult)
ordered_neurons_conn = [sortperm(weighted_tv_conn[adult_subnetid][1][id]) for id in 1:num_res] 
conn_mem_eg_adult = Vector{Any}(undef, length(1:2:size(conn_ESNs[adult_subnetid][1], 1)))
@threads for i in 1:2:size(conn_ESNs[adult_subnetid][1], 1)
    cropped = [prune_nodes_from_network(conn_ESNs[adult_subnetid][r], ordered_neurons_conn[r][1:i]) for r in 1:num_res]
    perf, _ = ReservoirTasks.res_performance_memory(cropped, 2, input_conn[adult_subnetid,1], reg_conn[adult_subnetid,1], leak_conn[adult_subnetid,1])
    conn_mem_eg_adult[i ÷ 2 + 1] = mean(perf)/baseline
end
            
starting_perf_er, _ = ReservoirTasks.res_performance_memory(er_ESNs[adult_subnetid][1:numres], 2, input_er[adult_subnetid,1], reg_er[adult_subnetid,1], leak_er[adult_subnetid,1])
baseline = mean(starting_perf_er)
ordered_neurons_er = [sortperm(weighted_tv_rand[adult_subnetid][1][id]) for id in 1:num_res]
er_mem_eg_adult = Vector{Any}(undef, length(1:2:size(er_ESNs[adult_subnetid][1], 1)))
@threads for i in 1:2:size(er_ESNs[adult_subnetid][1], 1)
    cropped = [prune_nodes_from_network(er_ESNs[adult_subnetid][r], ordered_neurons_er[r][1:i]) for r in 1:num_res]
    perf, _ = ReservoirTasks.res_performance_memory(cropped, 2, input_er[adult_subnetid,1], reg_er[adult_subnetid,1], leak_er[adult_subnetid,1])
    er_mem_eg_adult[i ÷ 2 + 1] = mean(perf)/baseline
end

p1 = plot(1/length(conn_mem_eg ./ conn_mem_eg[1]):1/length(conn_mem_eg ./ conn_mem_eg[1]):1,conn_mem_eg ./ conn_mem_eg[1], c=:dodgerblue4, lw=4, xlabel="Number of Neurons Pruned", ylabel="Relative Performance", grid=false, title="Pruning Example: Memory Task", legend=false)
plot!(1/length(conn_mem_eg ./ conn_mem_eg[1]):1/length(conn_mem_eg ./ conn_mem_eg[1]):1, er_mem_eg ./ er_mem_eg[1], c=:crimson, lw=4, xlabel="Number of Neurons Pruned", size=(400,400), ylabel="Relative Performance", title="Pruning Example: Memory Task", legend=false)
plot!(1/length(conn_mem_eg_adult ./ conn_mem_eg_adult[1]):1/length(conn_mem_eg_adult ./ conn_mem_eg_adult[1]):1, conn_mem_eg_adult ./ conn_mem_eg_adult[1], c=dodgerblues[3], lw=4, xlabel="Number of Neurons Pruned", ylabel="Relative Performance", grid=false, title="Pruning Example: Memory Task", legend=false)
plot!(1/length(er_mem_eg_adult ./ er_mem_eg_adult[1]):1/length(er_mem_eg_adult ./ er_mem_eg_adult[1]):1, er_mem_eg_adult ./ er_mem_eg_adult[1], c=crimsons[3], lw=4, xlabel="Number of Neurons Pruned", size=(400,400), ylabel="Relative Performance", title="Pruning Example: Memory Task", legend=false)



# DECISION MAKING

biases = vcat(reverse(0.41:0.01:1.0), reverse(0.1:0.1:0.4))

num_res = 30
trials_no = 6

larva_subnetid = 11
perfs_dm_conn = []
perfs_dm_er = []
ordered_neurons_conn = [sortperm(weighted_tv_conn[larva_subnetid][3][id]) for id in 1:num_res]
cuts = 1:2:size(conn_ESNs[larva_subnetid][1], 1)
@progress for i in cuts
    cropped = [prune_nodes_from_network(conn_ESNs[larva_subnetid][r], ordered_neurons_conn[r][1:i]) for r in 1:num_res]
    perf = ReservoirTasks.res_performance_decisionmaking(cropped, trials_no, input_conn[larva_subnetid,3], reg_conn[larva_subnetid,3], leak_conn[larva_subnetid,3], biases; threshold=0.8)
    perf = vcat(perf...)
    mean(perf)
    push!(perfs_dm_conn, mean(perf))
end
ordered_neurons_er = [sortperm(weighted_tv_rand[larva_subnetid][3][id]) for id in 1:num_res]
@progress for i in cuts
    cropped = [prune_nodes_from_network(er_ESNs[larva_subnetid][r], ordered_neurons_er[r][1:i]) for r in 1:num_res]
    perf = ReservoirTasks.res_performance_decisionmaking(cropped, trials_no, input_er[larva_subnetid,3], reg_er[larva_subnetid,3], leak_er[larva_subnetid,3], biases; threshold=0.8)
    perf = vcat(perf...)
    mean(perf)
    push!(perfs_dm_er, mean(perf))
end

p2 = plot(1/length(perfs_dm_conn):1/length(perfs_dm_conn):1, perfs_dm_conn ./ mean(perfs_dm_conn[1:5]), c=:dodgerblue4, lw=4, ylabel="Relative Performance", grid=false, title="Pruning Example: Decision Making Task", legend=false)
plot!(1/length(perfs_dm_er):1/length(perfs_dm_er):1, perfs_dm_er ./ mean(perfs_dm_er[1:20]), c=:crimson, lw=4, xlabel="Number of Neurons Pruned", size=(400,400), ylabel="Relative Performance", title="Pruning Example: Decision Making Task", legend=false)
plot!(1/length(perfs_dm_conn_adult):1/length(perfs_dm_conn_adult):1, perfs_dm_conn_adult ./ mean(perfs_dm_conn_adult[1:20]), c=dodgerblues[3], lw=4, ylabel="Relative Performance", grid=false, title="$adult_subnetid", legend=false)
plot!(1/length(perfs_dm_er_adult):1/length(perfs_dm_er_adult):1, perfs_dm_er_adult ./ mean(perfs_dm_er_adult[1:20]), lw=4, c=crimsons[3], size=(400,400))

# can do moving average to smooth out the curves a bit


adult_subnetid = 15

perfs_dm_conn_adult = []
perfs_dm_er_adult = []

cuts = 1:3:size(conn_ESNs[adult_subnetid][1], 1)
ordered_neurons_conn_adult = [sortperm(weighted_tv_conn[adult_subnetid][3][id]) for id in 1:num_res]
@progress for i in cuts
    cropped = [prune_nodes_from_network(conn_ESNs[adult_subnetid][r], ordered_neurons_conn_adult[r][1:i]) for r in 1:num_res]
    perf = ReservoirTasks.res_performance_decisionmaking(cropped, trials_no, input_conn[adult_subnetid,3], reg_conn[adult_subnetid,3], leak_conn[adult_subnetid,3], biases; threshold=0.8)
    perf = vcat(perf...)
    mean(perf)
    push!(perfs_dm_conn_adult, mean(perf))
end
ordered_neurons_er_adult = [sortperm(weighted_tv_rand[adult_subnetid][3][id]) for id in 1:num_res]
@progress for i in cuts
    cropped = [prune_nodes_from_network(er_ESNs[adult_subnetid][r], ordered_neurons_er_adult[r][1:i]) for r in 1:num_res]
    perf = ReservoirTasks.res_performance_decisionmaking(cropped, trials_no, input_er[adult_subnetid,3], reg_er[adult_subnetid,3], leak_er[adult_subnetid,3], biases; threshold=0.8)
    perf = vcat(perf...)
    mean(perf)
    push!(perfs_dm_er_adult, mean(perf))
end


p2 = plot(1/length(perfs_dm_conn):1/length(perfs_dm_conn):1, perfs_dm_conn ./ mean(perfs_dm_conn[1:20]), c=:dodgerblue4, lw=4, ylabel="Relative Performance", grid=false, title="Pruning Example: Decision Making Task", legend=false)
plot!(1/length(perfs_dm_er):1/length(perfs_dm_er):1, perfs_dm_er ./ mean(perfs_dm_er[1:20]), c=:crimson, lw=4, xlabel="Number of Neurons Pruned", size=(400,400), ylabel="Relative Performance", title="Pruning Example: Decision Making Task", legend=false)
plot!(1/length(perfs_dm_conn_adult):1/length(perfs_dm_conn_adult):1, perfs_dm_conn_adult ./ mean(perfs_dm_conn_adult[1:20]), c=dodgerblues[3], lw=4, ylabel="Relative Performance", grid=false, title="$adult_subnetid", legend=false)
    plot!(1/length(perfs_dm_er_adult):1/length(perfs_dm_er_adult):1, perfs_dm_er_adult ./ mean(perfs_dm_er_adult[1:20]), lw=4, c=crimsons[3], size=(400,400))

# can do moving average to smooth out the curves a bit








###############################################################################################################
# OSCILLATOR

subnetid = 10
num_res = 30
trials_no = 10

starting_perf_conn, _ = ReservoirTasks.res_performance_oscillator(conn_ESNs[subnetid][1:numres], osc_data, 0.5, trials_no, input_conn[subnetid,5], reg_conn[subnetid,5], leak_conn[subnetid,5])
baseline = mean(starting_perf_conn)
ordered_neurons_conn = [sortperm(weighted_tv_conn[subnetid][5][id]) for id in 1:num_res] 
conn_osc_eg = Vector{Any}(undef, length(1:2:size(conn_ESNs[subnetid][1], 1)))
@threads for i in 1:2:size(conn_ESNs[subnetid][1], 1)
    cropped = [prune_nodes_from_network(conn_ESNs[subnetid][r], ordered_neurons_conn[r][1:i]) for r in 1:num_res]
    perf = ReservoirTasks.res_performance_oscillator(cropped, osc_data, 0.5, trials_no, input_conn[subnetid,5], reg_conn[subnetid,5], leak_conn[subnetid,5])
    conn_osc_eg[i ÷ 2 + 1] = mean(mean(perf)/baseline)
end

starting_perf_er, _ = ReservoirTasks.res_performance_oscillator(er_ESNs[subnetid][1:numres], osc_data, 0.5, trials_no, input_er[subnetid,5], reg_er[subnetid,5], leak_er[subnetid,5])
baseline = mean(starting_perf_er)
ordered_neurons_er = [sortperm(weighted_tv_rand[subnetid][5][id]) for id in 1:num_res]
er_osc_eg = Vector{Any}(undef, length(1:2:size(er_ESNs[subnetid][1], 1)))
@threads for i in 1:2:size(er_ESNs[subnetid][1], 1)
    cropped = [prune_nodes_from_network(er_ESNs[subnetid][r], ordered_neurons_er[r][1:i]) for r in 1:num_res]
    perf = ReservoirTasks.res_performance_oscillator(cropped, osc_data, 0.5, trials_no, input_er[subnetid,5], reg_er[subnetid,5], leak_er[subnetid,5])
    er_osc_eg[i ÷ 2 + 1] = mean(mean(perf)/baseline)
end

subnetid_adult = 30

starting_perf_conn_adult, _ = ReservoirTasks.res_performance_oscillator(conn_ESNs[subnetid_adult][1:numres], osc_data, 0.5, trials_no, input_conn[subnetid_adult,5], reg_conn[subnetid_adult,5], leak_conn[subnetid_adult,5])
baseline = mean(starting_perf_conn_adult)
ordered_neurons_conn = [sortperm(weighted_tv_conn[subnetid_adult][5][id]) for id in 1:num_res]
conn_osc_eg_adult = Vector{Any}(undef, length(1:2:size(conn_ESNs[subnetid_adult][1], 1)))
@threads for i in 1:2:size(conn_ESNs[subnetid_adult][1], 1)
    cropped = [prune_nodes_from_network(conn_ESNs[subnetid_adult][r], ordered_neurons_conn[r][1:i]) for r in 1:num_res]
    perf = ReservoirTasks.res_performance_oscillator(cropped, osc_data, 0.5, trials_no, input_conn[subnetid_adult,5], reg_conn[subnetid_adult,5], leak_conn[subnetid_adult,5])
    conn_osc_eg_adult[i ÷ 2 + 1] = mean(mean(perf)/baseline)
end

starting_perf_er_adult, _ = ReservoirTasks.res_performance_oscillator(er_ESNs[subnetid_adult][1:numres], osc_data, 0.5, trials_no, input_er[subnetid_adult,5], reg_er[subnetid_adult,5], leak_er[subnetid_adult,5])
baseline = mean(starting_perf_er_adult)
ordered_neurons_er = [sortperm(weighted_tv_rand[subnetid_adult][5][id]) for id in 1:num_res]
er_osc_eg_adult = Vector{Any}(undef, length(1:2:size(er_ESNs[subnetid_adult][1], 1)))
@threads for i in 1:2:size(er_ESNs[subnetid_adult][1], 1)
    cropped = [prune_nodes_from_network(er_ESNs[subnetid_adult][r], ordered_neurons_er[r][1:i]) for r in 1:num_res]
    perf = ReservoirTasks.res_performance_oscillator(cropped, osc_data, 0.5, trials_no, input_er[subnetid_adult,5], reg_er[subnetid_adult,5], leak_er[subnetid_adult,5])
    er_osc_eg_adult[i ÷ 2 + 1] = mean(mean(perf)/baseline)
end


v1 = vec(mean(hcat(conn_osc_eg,conn_osc_eg1),dims=2))
v2 = vec(mean(hcat(er_osc_eg,er_osc_eg1),dims=2))
v3 = vec(mean(hcat(conn_osc_eg_adult,conn_osc_eg_adult1),dims=2))
v4 = vec(mean(hcat(er_osc_eg_adult,er_osc_eg_adult1),dims=2))


p3 = plot(1/length(v1):1/length(v1):1, v1 ./ mean(v1[1:10]), c=:dodgerblue4, lw=4, xlabel="Number of Neurons Pruned", ylabel="Relative Performance", grid=false, title="Pruning Example: Oscillator Task", legend=false)
plot!(1/length(v2):1/length(v2):1, v2 ./ mean(v2[1:10]), c=:crimson, lw=4, xlabel="Number of Neurons Pruned", size=(400,400), ylabel="Relative Performance", title="Pruning Example: Oscillator Task", legend=false)
plot!(1/length(v3):1/length(v3):1, v3 ./ mean(v3[1:10]), c=dodgerblues[3], lw=4, xlabel="Number of Neurons Pruned", ylabel="Relative Performance", grid=false, title="Pruning Example: Oscillator Task", legend=false)
plot!(1/length(v4):1/length(v4):1, v4 ./ mean(v4[1:10]), c=crimsons[3], lw=4, xlabel="Number of Neurons Pruned", size=(400,400), ylabel="Relative Performance", title="Pruning Example: Oscillator Task", legend=false)

# can do moving average to smooth out the curves a bit




