using LightGraphs, Graphs
using GraphRecipes


# Scatter plot of correlations between node degree and WTV
lenn=30

asps_conn = [[Graphs.local_clustering_coefficient(Graphs.SimpleDiGraph(deepcopy((conn_ESNs[i][j])))) for j in 1:lenn] for i in 1:39]
asps_er = [[Graphs.local_clustering_coefficient(Graphs.SimpleDiGraph(deepcopy((er_ESNs[i][j])))) for j in 1:lenn] for i in 1:39]
asps_cfg = [[Graphs.local_clustering_coefficient(Graphs.SimpleDiGraph(deepcopy((cfg_ESNs[i][j])))) for j in 1:lenn] for i in 1:39]


mean_conn_asps = [mean(asps_conn[i][1]) for i in 1:39]
mean_er_asps = [mean(asps_er[i][1]) for i in 1:39]
mean_cfg_asps = [mean(asps_cfg[i][1]) for i in 1:39]


pvalue(SignedRankTest(mean_conn_asps[3:11], mean_er_asps[3:11]))
pvalue(SignedRankTest(mean_conn_asps[12:end], mean_er_asps[12:end]))


s = scatter(jitter(1, 28), mean_conn_asps[12:end], c=dodgerblues[3], markerstrokewidth=0, label=false)
scatter!(jitter(2, 28), mean_er_asps[12:end], c=crimsons[3], markerstrokewidth=0, label=false)
scatter!(jitter(3, 28), mean_cfg_asps[12:end], c=oranges[3], markerstrokewidth=0, label=false)
scatter!(jitter(1, 9), mean_conn_asps[3:11], c=dodgerblues[2], markerstrokewidth=0, label=false)
scatter!(jitter(2, 9), mean_er_asps[3:11], c=crimsons[2], markerstrokewidth=0, label=false)
scatter!(jitter(3, 9), mean_cfg_asps[3:11], c=oranges[2], markerstrokewidth=0, label=false)
plot!(title="Local Clustering Coefficient", size=(500,300), grid=false, xticks=([1,2,3], ["Conn","ER","CFG"]), ylabel="Mean Local Clustering Coefficient", ylim=(0,0.05))

using Statistics
s = bar([1],[mean(mean_conn_asps[3:11])], c=:dodgerblue4, label="Conn", yerror=std(mean_conn_asps[3:11])/sqrt(9), alpha=0.8)
bar!([2],[mean(mean_er_asps[3:11])], c=:crimson, label="Random", yerror=std(mean_er_asps[3:11])/sqrt(9), alpha=0.8)
bar!([3],[mean(mean_cfg_asps[3:11])], c=:orange, label="CFG", yerror=std(mean_cfg_asps[3:11])/sqrt(9), alpha=0.8)
bar!([5],[mean(mean_conn_asps[12:end])], c=dodgerblues[3], label=false, yerror=std(mean_conn_asps[12:end])/sqrt(28))
bar!([6],[mean(mean_er_asps[12:end])], c=crimsons[3], label=false, yerror=std(mean_er_asps[12:end])/sqrt(28))
bar!([7],[mean(mean_cfg_asps[12:end])], c=oranges[3], label=false, yerror=std(mean_cfg_asps[12:end])/sqrt(28))
plot!(ylabel="Mean Local \n Clustering Coefficient \n", xticks=([2,6],["Larva", "Adult"]), title="Local Clustering Coefficient", ylim=(0,0.02), yticks=0:0.01:0.02, size=(500,300), grid=false)
plot!(ylabelfontsize=14, ytickfontsize=12, titlefontsize=16, legendfontsize=10, xtickfontsize=12, leftmargin=5mm)



# Compute correlations for each subnetwork and each task

asp_cor_task1_conn = mean.([[cor(asps_conn[j][i],weighted_tv_conn[j][1][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task2_conn = mean.([[cor(asps_conn[j][i],weighted_tv_conn[j][2][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task3_conn = mean.([[cor(asps_conn[j][i],weighted_tv_conn[j][3][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task4_conn = mean.([[cor(asps_conn[j][i],weighted_tv_conn[j][4][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task5_conn = mean.([[cor(asps_conn[j][i],weighted_tv_conn[j][5][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task6_conn = mean.([[cor(asps_conn[j][i],weighted_tv_conn[j][6][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task7_conn = mean.([[cor(asps_conn[j][i],weighted_tv_conn[j][7][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task8_conn = mean.([[cor(asps_conn[j][i],weighted_tv_conn[j][8][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task1_er =  mean.([[cor(asps_er[j][i],weighted_tv_rand[j][1][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task2_er =  mean.([[cor(asps_er[j][i],weighted_tv_rand[j][2][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task3_er =  mean.([[cor(asps_er[j][i],weighted_tv_rand[j][3][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task4_er =  mean.([[cor(asps_er[j][i],weighted_tv_rand[j][4][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task5_er =  mean.([[cor(asps_er[j][i],weighted_tv_rand[j][5][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task6_er =  mean.([[cor(asps_er[j][i],weighted_tv_rand[j][6][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task7_er =  mean.([[cor(asps_er[j][i],weighted_tv_rand[j][7][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task8_er =  mean.([[cor(asps_er[j][i],weighted_tv_rand[j][8][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task1_cfg =  mean.([[cor(asps_cfg[j][i],weighted_tv_cfg[j][1][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task2_cfg =  mean.([[cor(asps_cfg[j][i],weighted_tv_cfg[j][2][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task3_cfg =  mean.([[cor(asps_cfg[j][i],weighted_tv_cfg[j][3][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task4_cfg =  mean.([[cor(asps_cfg[j][i],weighted_tv_cfg[j][4][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task5_cfg =  mean.([[cor(asps_cfg[j][i],weighted_tv_cfg[j][5][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task6_cfg =  mean.([[cor(asps_cfg[j][i],weighted_tv_cfg[j][6][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task7_cfg =  mean.([[cor(asps_cfg[j][i],weighted_tv_cfg[j][7][i]) for i in 1:lenn] for j in 3:39])
asp_cor_task8_cfg =  mean.([[cor(asps_cfg[j][i],weighted_tv_cfg[j][8][i]) for i in 1:lenn] for j in 3:39])


asp_conns = [asp_cor_task1_conn, asp_cor_task2_conn, asp_cor_task3_conn, asp_cor_task4_conn, asp_cor_task5_conn, asp_cor_task6_conn, asp_cor_task7_conn, asp_cor_task8_conn]
asp_ers = [asp_cor_task1_er, asp_cor_task2_er, asp_cor_task3_er, asp_cor_task4_er, asp_cor_task5_er, asp_cor_task6_er, asp_cor_task7_er, asp_cor_task8_er]
asp_cfgs = [asp_cor_task1_cfg, asp_cor_task2_cfg, asp_cor_task3_cfg, asp_cor_task4_cfg, asp_cor_task5_cfg, asp_cor_task6_cfg, asp_cor_task7_cfg, asp_cor_task8_cfg]

p4 = plot()

# Define the base x-positions and the number of data columns to plot
x_bases = [1, 5, 9, 13, 17, 21, 25, 29] # Base x for 'conn' data for each task
num_tasks = 8

# Loop through each task/column of data
for i in 1:num_tasks
    col = i
    x_base = x_bases[i]

    # # Group 3 (rows 12:end)
    n_end = length(12:39) # Safely get number of points
    scatter!(p4, jitter(x_base, n_end), asp_conns[i][10:end], c=dodgerblues[3], markerstrokewidth=0, label=false)
    scatter!(p4, jitter(x_base + 1, n_end), asp_ers[i][10:end], c=crimsons[3], markerstrokewidth=0, label=false)
    # scatter!(p4, jitter(x_base + 2, n_end), asp_cfgs[i][10:end], c=oranges[3], markerstrokewidth=0, label=false)

    
    # Group 2 (rows 3:11)
    scatter!(p4, jitter(x_base, 9), asp_conns[i][1:9], c=:dodgerblue4, markerstrokewidth=0, label=false)
    scatter!(p4, jitter(x_base + 1, 9), asp_ers[i][1:9], c=:crimson, markerstrokewidth=0, label=false)
    # scatter!(p4, jitter(x_base + 2, 9), asp_cfgs[i][1:9], c=:orange, markerstrokewidth=0, label=false)

end

# Add vertical lines and final plot settings
vline!(p4, [4, 8, 12, 16, 20, 24, 28] .- 0.5, c=:black, linestyle=:dash, label=false, grid=false)
hline!(p4, [0.0], c=:grey, lw=3, label=false)
plot!(p4, xticks=([2, 6, 10, 14, 18, 22, 26, 30] .- 0.5, ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8"]), 
title="WTV and Local Clustering Correlation",ylim=(-0.5,0.5))

using Plots.PlotMeasures

pp4 = plot(p1,p2,p4, layout=(3,1),size=(700,1200),leftmargin=10mm,dpi=600)
