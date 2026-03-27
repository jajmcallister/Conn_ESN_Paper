

Ls = collect(10:5:100); pushfirst!(Ls, 1:9...)
biases = vcat(reverse(0.55:0.05:1.0), reverse(0.1:0.02:0.5), reverse(0.0:0.005:0.99))
variances = vcat(0.01:0.01,0.09, 0.1:0.2:1., 1.1:0.1:1.5, 1.55:0.05:4)


conn_energy_memory = Vector{Any}(undef, 39)
er_energy_memory = Vector{Any}(undef, 39)
conn_energy_recall = Vector{Any}(undef, 39)
er_energy_recall = Vector{Any}(undef, 39)
conn_energy_dm = Vector{Any}(undef, 39)
er_energy_dm = Vector{Any}(undef, 39)
conn_energy_ddm = Vector{Any}(undef, 39)
er_energy_ddm = Vector{Any}(undef, 39)
conn_energy_osc = Vector{Any}(undef, 39)
er_energy_osc = Vector{Any}(undef, 39)
conn_energy_lot = Vector{Any}(undef, 39)
er_energy_lot = Vector{Any}(undef, 39)
conn_energy_lor = Vector{Any}(undef, 39)
er_energy_lor = Vector{Any}(undef, 39)
conn_energy_ros = Vector{Any}(undef, 39)
er_energy_ros = Vector{Any}(undef, 39)

using Base.Threads
bot = 3
top = 39
numres= 5

@threads for id in bot:top
    _, conn_energy_memory1 = ReservoirTasks.res_performance_memory_wout(conn_ESNs[id][1:numres],2,input_conn[id,1], reg_conn[id,1], leak_conn[id,1])
    _, er_energy_memory1 = ReservoirTasks.res_performance_memory_wout(er_ESNs[id][1:numres],2,input_conn[id,1], reg_conn[id,1], leak_conn[id,1])
    _, conn_energy_recall1 = ReservoirTasks.res_performance_recall_wout(conn_ESNs[id][1:numres], 2, input_conn[id,2], reg_conn[id,2], leak_conn[id,2], Ls; threshold=0.8)
    _, er_energy_recall1 = ReservoirTasks.res_performance_recall_wout(er_ESNs[id][1:numres], 2, input_conn[id,2], reg_conn[id,2], leak_conn[id,2], Ls; threshold=0.8)
    _, conn_energy_dm1 = ReservoirTasks.res_performance_decisionmaking_wout(conn_ESNs[id][1:numres], 2, input_conn[id,3], reg_conn[id,3], leak_conn[id,3], biases; threshold=0.8)
    _, er_energy_dm1 = ReservoirTasks.res_performance_decisionmaking_wout(er_ESNs[id][1:numres], 2, input_conn[id,3], reg_conn[id,3], leak_conn[id,3], biases; threshold=0.8)
    _, conn_energy_ddm1 = ReservoirTasks.res_performance_delay_decisionmaking_wout(conn_ESNs[id][1:numres], 2, input_conn[id,4], reg_conn[id,4], leak_conn[id,4], variances; threshold=0.8)
    _, er_energy_ddm1 = ReservoirTasks.res_performance_delay_decisionmaking_wout(er_ESNs[id][1:numres], 2, input_conn[id,4], reg_conn[id,4], leak_conn[id,4], variances; threshold=0.8)
    _, conn_energy_osc1 = ReservoirTasks.res_performance_oscillator_wout(conn_ESNs[id][1:numres], osc_data, 0.5, 2, input_conn[id,5], reg_conn[id,5], leak_conn[id,5])
    _, er_energy_osc1 = ReservoirTasks.res_performance_oscillator_wout(er_ESNs[id][1:numres], osc_data, 0.5, 2, input_conn[id,5], reg_conn[id,5], leak_conn[id,5])
    _, conn_energy_lot1 = ReservoirTasks.res_performance_lotka_2d_wout(conn_ESNs[id][1:numres], lotka_data, 0.5, 2, input_conn[id,6], reg_conn[id,6], leak_conn[id,6])
    _, er_energy_lot1 = ReservoirTasks.res_performance_lotka_2d_wout(er_ESNs[id][1:numres], lotka_data, 0.5, 2, input_conn[id,6], reg_conn[id,6], leak_conn[id,6])
    _, conn_energy_lor1 = ReservoirTasks.res_performance_lorenz_wout(conn_ESNs[id][1:numres], lorenz_train_data, lorenz_test_data, 2, 0.5, input_conn[id,7], reg_conn[id,7], leak_conn[id,7])
    _, er_energy_lor1 = ReservoirTasks.res_performance_lorenz_wout(er_ESNs[id][1:numres], lorenz_train_data, lorenz_test_data, 2, 0.5, input_conn[id,7], reg_conn[id,7], leak_conn[id,7])
    _, conn_energy_ros1 = ReservoirTasks.res_performance_rossler_wout(conn_ESNs[id][1:numres], rossler_train_data, rossler_test_data, 2, 0.5, input_conn[id,8], reg_conn[id,8], leak_conn[id,8])
    _, er_energy_ros1 = ReservoirTasks.res_performance_rossler_wout(er_ESNs[id][1:numres], rossler_train_data, rossler_test_data, 2, 0.5, input_conn[id,8], reg_conn[id,8], leak_conn[id,8])

    conn_energy_memory[id] = vcat(conn_energy_memory1...)
    er_energy_memory[id] = vcat(er_energy_memory1...)
    conn_energy_recall[id] = vcat(conn_energy_recall1...)
    er_energy_recall[id] = vcat(er_energy_recall1...)
    conn_energy_dm[id] = vcat(conn_energy_dm1...)
    er_energy_dm[id] = vcat(er_energy_dm1...)
    conn_energy_ddm[id] = vcat(conn_energy_ddm1...)
    er_energy_ddm[id] = vcat(er_energy_ddm1...)
    conn_energy_osc[id] = vcat(conn_energy_osc1...)
    er_energy_osc[id] = vcat(er_energy_osc1...)
    conn_energy_lot[id] = vcat(conn_energy_lot1...)
    er_energy_lot[id] = vcat(er_energy_lot1...)
    conn_energy_lor[id] = vcat(conn_energy_lor1...)
    er_energy_lor[id] = vcat(er_energy_lor1...)
    conn_energy_ros[id] = vcat(conn_energy_ros1...)
    er_energy_ros[id] = vcat(er_energy_ros1...)
end

B1 = [filter(!isnothing, subarray) for subarray in conn_energy_recall[3:39]]
B2 = [filter(!isnothing, subarray) for subarray in er_energy_recall[3:39]]
C1 = [filter(!isnothing, subarray) for subarray in conn_energy_dm[3:39]]
C2 = [filter(!isnothing, subarray) for subarray in er_energy_dm[3:39]]
D1 = [filter(!isnothing, subarray) for subarray in conn_energy_ddm[3:39]]
D2 = [filter(!isnothing, subarray) for subarray in er_energy_ddm[3:39]]


conn_en = [conn_energy_memory[bot:top],
            B1,
            C1,
            D1,
            conn_energy_osc[bot:top],
            conn_energy_lot[bot:top],
            conn_energy_lor[bot:top],
            conn_energy_ros[bot:top]]
er_en = [er_energy_memory[bot:top],
            B2,
            C2,
            D2,
            er_energy_osc[bot:top],
            er_energy_lot[bot:top],
            er_energy_lor[bot:top],
            er_energy_ros[bot:top]]


sizess = [size(conn_ESNs[i][1])[2] for i in bot:top]

conn_energies = [mean.(conn_energy_memory[bot:top]) ./ sizess,
                    mean.(B1) ./ sizess,
                    mean.(C1) ./ sizess,
                    mean.(D1) ./ sizess,
                    mean.(conn_energy_osc[bot:top]) ./ sizess,
                    mean.(conn_energy_lot[bot:top]) ./ sizess,
                    mean.(conn_energy_lor[bot:top]) ./ sizess,
                    mean.(conn_energy_ros[bot:top]) ./ sizess]
er_energies = [mean.(er_energy_memory[bot:top]) ./ sizess,
                    mean.(B2) ./ sizess,
                    mean.(C2) ./ sizess,
                    mean.(D2) ./ sizess,
                    mean.(er_energy_osc[bot:top]) ./ sizess,
                    mean.(er_energy_lot[bot:top]) ./ sizess,
                    mean.(er_energy_lor[bot:top]) ./ sizess,
                    mean.(er_energy_ros[bot:top]) ./ sizess]

conn_energies = [mean.(conn_energy_memory[bot:top]),
                    mean.(B1),
                    mean.(C1),
                    mean.(D1),
                    mean.(conn_energy_osc[bot:top]),
                    mean.(conn_energy_lot[bot:top]),
                    mean.(conn_energy_lor[bot:top]),
                    mean.(conn_energy_ros[bot:top])]
er_energies = [mean.(er_energy_memory[bot:top]),
                    mean.(B2),
                    mean.(C2),
                    mean.(D2),
                    mean.(er_energy_osc[bot:top]),
                    mean.(er_energy_lot[bot:top]),
                    mean.(er_energy_lor[bot:top]),
                    mean.(er_energy_ros[bot:top])]


b1 = bar([1], [mean(conn_energies[1][1:9])],yerror = std(conn_energies[1][1:9]) / sqrt(9), lw=3,color=:dodgerblue4, alpha=0.8)
bar!([1.5], [mean(conn_energies[1][10:end])], yerror = std(conn_energies[1][10:end]) / sqrt(28), lw=3,color=dodgerblues[3], alpha=0.8)
bar!([3], [mean(er_energies[1][1:9])], yerror = std(er_energies[1][1:9]) / sqrt(9), lw=3,color=:crimson, alpha=0.8)
bar!([3.5], [mean(er_energies[1][10:end])], yerror = std(er_energies[1][10:end]) / sqrt(28), lw=3, color=crimsons[3],alpha=0.8)

b2 = bar([1], [mean(conn_energies[2][1:9])], yerror = std(conn_energies[2][1:9]) / sqrt(9),lw=3, color=:dodgerblue4, alpha=0.8)
bar!([1.5], [mean(conn_energies[2][10:end])], yerror = std(conn_energies[2][10:end]) / sqrt(28), lw=3,color=dodgerblues[3], alpha=0.8)
bar!([3], [mean(er_energies[2][1:9])], yerror = std(er_energies[2][1:9]) / sqrt(9), lw=3,color=:crimson, alpha=0.8)
bar!([3.5], [mean(er_energies[2][10:end])], yerror = std(er_energies[2][10:end]) / sqrt(28), lw=3, color=crimsons[3], alpha=0.8)

b3 = bar([1], [mean(conn_energies[3][1:9])], yerror = std(conn_energies[3][1:9]) / sqrt(9), lw=3,color=:dodgerblue4, alpha=0.8)
bar!([1.5], [mean(conn_energies[3][10:end])], yerror = std(conn_energies[3][10:end]) / sqrt(28), lw=3,color=dodgerblues[3], alpha=0.8)
bar!([3], [mean(er_energies[3][1:9])], yerror = std(er_energies[3][1:9]) / sqrt(9), lw=3,color=:crimson, alpha=0.8)
bar!([3.5], [mean(er_energies[3][10:end])], yerror = std(er_energies[3][10:end]) / sqrt(28), lw=3, color=crimsons[3], alpha=0.8)

b4 = bar([1], [mean(conn_energies[4][1:9])], yerror = std(conn_energies[4][1:9]) / sqrt(9), lw=3,color=:dodgerblue4, alpha=0.8)
bar!([1.5], [mean(conn_energies[4][10:end])], yerror = std(conn_energies[4][10:end]) / sqrt(28), lw=3,color=dodgerblues[3], alpha=0.8)
bar!([3], [mean(er_energies[4][1:9])], yerror = std(er_energies[4][1:9]) / sqrt(9), lw=3,color=:crimson, alpha=0.8)
bar!([3.5], [mean(er_energies[4][10:end])], yerror = std(er_energies[4][10:end]) / sqrt(28),lw=3, color=crimsons[3], alpha=0.8)

b5 = bar([1], [mean(conn_energies[5][1:9])], yerror = std(conn_energies[5][1:9]) / sqrt(9), lw=3,color=:dodgerblue4, alpha=0.8)
bar!([1.5], [mean(conn_energies[5][10:end])], yerror = std(conn_energies[5][10:end]) / sqrt(28), lw=3,color=dodgerblues[3], alpha=0.8)
bar!([3], [mean(er_energies[5][1:9])], yerror = std(er_energies[5][1:9]) / sqrt(9), lw=3,color=:crimson, alpha=0.8)
bar!([3.5], [mean(er_energies[5][10:end])], yerror = std(er_energies[5][10:end]) / sqrt(28), lw=3, color=crimsons[3], alpha=0.8)

b6 = bar([1], [mean(conn_energies[6][1:9])], yerror = std(conn_energies[6][1:9]) / sqrt(9), lw=3,color=:dodgerblue4, alpha=0.8)
bar!([1.5], [mean(conn_energies[6][10:end])], yerror = std(conn_energies[6][10:end]) / sqrt(28), lw=3,color=dodgerblues[3], alpha=0.8)
bar!([3], [mean(er_energies[6][1:9])], yerror = std(er_energies[6][1:9]) / sqrt(9), lw=3,color=:crimson, alpha=0.8)
bar!([3.5], [mean(er_energies[6][10:end])], yerror = std(er_energies[6][10:end]) / sqrt(28), lw=3,color=crimsons[3],alpha=0.8)

b7 = bar([1], [mean(conn_energies[7][1:9])], yerror = std(conn_energies[7][1:9]) / sqrt(9), lw=3,color=:dodgerblue4, alpha=0.8)
bar!([1.5], [mean(conn_energies[7][10:end])], yerror = std(conn_energies[7][10:end]) / sqrt(28), lw=3,color=dodgerblues[3], alpha=0.8)
bar!([3], [mean(er_energies[7][1:9])], yerror = std(er_energies[7][1:9]) / sqrt(9), lw=3,color=:crimson, alpha=0.8)
bar!([3.5], [mean(er_energies[7][10:end])], yerror = std(er_energies[7][10:end]) / sqrt(28), lw=3, color=crimsons[3], alpha=0.8)

b8 = bar([1], [mean(conn_energies[8][1:9])], yerror = std(conn_energies[8][1:9]) / sqrt(9), lw=3,color=:dodgerblue4, alpha=0.8)
bar!([1.5], [mean(conn_energies[8][10:end])], yerror = std(conn_energies[8][10:end]) / sqrt(28), lw=3,color=dodgerblues[3], alpha=0.8)
bar!([3], [mean(er_energies[8][1:9])], yerror = std(er_energies[8][1:9]) / sqrt(9), lw=3,color=:crimson, alpha=0.8)
bar!([3.5], [mean(er_energies[8][10:end])], yerror = std(er_energies[8][10:end]) / sqrt(28),lw=3, color=crimsons[3], alpha=0.8)

p = plot(b1,b2,b3,b4,b5,b6,b7,b8,layout=(4,2), ylim=(0,0.1),size=(800,800), plot_title = "Mean Energy Cost per Task", legend=false, grid=false, xticks=false)


using HypothesisTests

pvalue(SignedRankTest(conn_energies[1][1:9], er_energies[1][1:9]))
pvalue(SignedRankTest(conn_energies[1][10:end], er_energies[1][10:end]))
pvalue(SignedRankTest(conn_energies[2][1:9], er_energies[2][1:9]))
pvalue(SignedRankTest(conn_energies[2][10:end], er_energies[2][10:end]))
pvalue(SignedRankTest(conn_energies[3][1:9], er_energies[3][1:9]))
pvalue(SignedRankTest(conn_energies[3][10:end], er_energies[3][10:end]))
pvalue(SignedRankTest(conn_energies[4][1:9], er_energies[4][1:9]))
pvalue(SignedRankTest(conn_energies[4][10:end], er_energies[4][10:end]))
pvalue(SignedRankTest(conn_energies[5][1:9], er_energies[5][1:9]))
pvalue(SignedRankTest(conn_energies[5][10:end], er_energies[5][10:end]))
pvalue(SignedRankTest(conn_energies[6][1:9], er_energies[6][1:9]))
pvalue(SignedRankTest(conn_energies[6][10:end], er_energies[6][10:end]))
pvalue(SignedRankTest(conn_energies[7][1:9], er_energies[7][1:9]))
pvalue(SignedRankTest(conn_energies[7][10:end], er_energies[7][10:end]))
pvalue(SignedRankTest(conn_energies[8][1:9], er_energies[8][1:9]))
pvalue(SignedRankTest(conn_energies[8][10:end], er_energies[8][10:end]))

