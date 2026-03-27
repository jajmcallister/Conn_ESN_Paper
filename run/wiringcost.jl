conn_esns = deepcopy(conn_ESNs)
rand_esns = deepcopy(er_ESNs)
cfg_esns = deepcopy(cfg_ESNs)

using .ConnectomeFunctions, .Reservoirs_src, .ReservoirTasks

conn_costs = [[sum(abs.(m)) for m in conn_esns[iid]] for iid in 1:39]
er_costs = [[sum(abs.(m)) for m in rand_esns[iid]] for iid in 1:39]
cfg_costs = [[sum(abs.(m)) for m in cfg_esns[iid]] for iid in 1:39]
conn_spars = [[compute_sparsity(m) for m in conn_esns[iid]] for iid in 1:39]
er_spars = [[compute_sparsity(m) for m in rand_esns[iid]] for iid in 1:39]
cfg_spars = [[compute_sparsity(m) for m in cfg_esns[iid]] for iid in 1:39]
conn_sizes = [[size(m, 1) for m in conn_esns[iid]] for iid in 1:39]
er_sizes = [[size(m, 1) for m in rand_esns[iid]] for iid in 1:39]
cfg_sizes = [[size(m, 1) for m in cfg_esns[iid]] for iid in 1:39]

using StatsPlots
boxplot([1], mean.(conn_costs))
boxplot!([2], mean.(er_costs))
boxplot!([3], mean.(cfg_costs))



jitter(x_base, n_points; width=0.3) = x_base .+ (rand(n_points) * 2 * width .- width)

# Create subplots - 2 rows, 4 columns for 8 tasks
num_tasks = 8

plot()
scatter!(jitter(x_positions[1], 39), mean.(conn_costs)[3:12], 
             c=dodgerblues[2], markerstrokewidth=0, label=false, markersize=3)
scatter!(jitter(x_positions[2], 39), mean.(er_costs)[3:12], 
             c=crimsons[2], markerstrokewidth=0, label=false, markersize=3)
scatter!(jitter(x_positions[3], 39), mean.(cfg_costs)[3:12], 
             c=oranges[2], markerstrokewidth=0, label=false, markersize=3)
scatter!(jitter(x_positions[1], 39), mean.(conn_costs)[13:end], 
             c=dodgerblues[3], markerstrokewidth=0, label=false, markersize=3)
scatter!(jitter(x_positions[2], 39), mean.(er_costs)[13:end], 
             c=crimsons[3], markerstrokewidth=0, label=false, markersize=3)
scatter!(jitter(x_positions[3], 39), mean.(cfg_costs)[13:end], 
             c=oranges[3], markerstrokewidth=0, label=false, markersize=3)


violin([1], mean.(conn_costs[3:12]),side=:left, color=dodgerblues[2], label=false)
violin!([1], mean.(conn_costs[13:end]), side=:right, color=dodgerblues[3], label=false)
violin!([2], mean.(er_costs[3:12]), side=:left, color=crimsons[2], label=false)
violin!([2], mean.(er_costs[13:end]), side=:right, color=crimsons[3], label=false)
violin!([3], mean.(cfg_costs[3:12]), side=:left, color=oranges[2], label=false)
violin!([3], mean.(cfg_costs[13:end]), side=:right, color=oranges[3], label=false)




# performances normalised by wiring cost

using Plots
using Colors

jitter(x_base, n_points; width=0.3) = x_base .+ (rand(n_points) * 2 * width .- width)

# Create subplots - 2 rows, 4 columns for 8 tasks
num_tasks = 8
plots_array = []

for i in 1:num_tasks
    col = i
    
    # Create individual subplot for this task
    p_task = plot(title="Task $i", titlefontsize=10)
    
    # Define x positions for the three network types
    x_positions = [1, 2, 3]
    
    # Group 3 (rows 13:end)
    n_end = length(13:size(mean_conn_perfs, 1))
    scatter!(p_task, jitter(x_positions[1], n_end), mean_conn_perfs[13:end, col] ./ mean.(conn_costs)[13:end], 
             c=dodgerblues[3], markerstrokewidth=0, label=false, markersize=3)
    scatter!(p_task, jitter(x_positions[2], n_end), mean_er_perfs[13:end, col] ./ mean.(er_costs)[13:end], 
             c=crimsons[3], markerstrokewidth=0, label=false, markersize=3)
    scatter!(p_task, jitter(x_positions[3], n_end), mean_cfg_perfs[13:end, col] ./ mean.(cfg_costs)[13:end], 
             c=oranges[3], markerstrokewidth=0, label=false, markersize=3)
    
    # Group 2 (rows 3:12)
    scatter!(p_task, jitter(x_positions[1], 10), mean_conn_perfs[3:12, col] ./ mean.(conn_costs)[3:12], 
             c=dodgerblues[2], markerstrokewidth=0, label=false, markersize=3)
    scatter!(p_task, jitter(x_positions[2], 10), mean_er_perfs[3:12, col] ./ mean.(er_costs)[3:12], 
             c=crimsons[2], markerstrokewidth=0, label=false, markersize=3)
    scatter!(p_task, jitter(x_positions[3], 10), mean_cfg_perfs[3:12, col] ./ mean.(cfg_costs)[3:12], 
             c=oranges[2], markerstrokewidth=0, label=false, markersize=3)
    
    # Group 1 (rows 1:2)
    scatter!(p_task, jitter(x_positions[1], 2), mean_conn_perfs[1:2, col] ./ mean.(conn_costs)[1:2], 
             c=dodgerblues[1], markerstrokewidth=0, label=false, markersize=3)
    scatter!(p_task, jitter(x_positions[2], 2), mean_er_perfs[1:2, col] ./ mean.(er_costs)[1:2], 
             c=crimsons[1], markerstrokewidth=0, label=false, markersize=3)
    scatter!(p_task, jitter(x_positions[3], 2), mean_cfg_perfs[1:2, col] ./ mean.(cfg_costs)[1:2], 
             c=oranges[1], markerstrokewidth=0, label=false, markersize=3)
    
    # Customize individual subplot
    plot!(p_task, 
          xticks=(x_positions, ["Conn", "ER", "CFG"]),
          ylabel="Performance",
          xlim=(0.5, 3.5),
          grid=false,
          tickfontsize=8,
          guidefontsize=8)
    
    push!(plots_array, p_task)
end

# Combine all subplots
combined_plot = plot(plots_array..., 
                    layout=(2, 4), 
                    size=(1200, 600),
                    margin=3Plots.mm)

# Create a separate legend plot
legend_plot = plot(showaxis=false, grid=false, legend=:top,
                  background_color_subplot=:transparent)

# Add invisible points just for legend
scatter!(legend_plot, [NaN], [NaN], c=dodgerblues[1], label="C. elegans", markerstrokewidth=0)
scatter!(legend_plot, [NaN], [NaN], c=crimsons[1], label="ER (C. elegans)", markerstrokewidth=0)
scatter!(legend_plot, [NaN], [NaN], c=oranges[1], label="CFG (C. elegans)", markerstrokewidth=0)

scatter!(legend_plot, [NaN], [NaN], c=dodgerblues[2], label="Drosophila Larva", markerstrokewidth=0)
scatter!(legend_plot, [NaN], [NaN], c=crimsons[2], label="ER (Drosophila Larva)", markerstrokewidth=0)
scatter!(legend_plot, [NaN], [NaN], c=oranges[2], label="CFG (Drosophila Larva)", markerstrokewidth=0)

scatter!(legend_plot, [NaN], [NaN], c=dodgerblues[3], label="Drosophila Adult", markerstrokewidth=0)
scatter!(legend_plot, [NaN], [NaN], c=crimsons[3], label="ER (Drosophila Adult)", markerstrokewidth=0)
scatter!(legend_plot, [NaN], [NaN], c=oranges[3], label="CFG (Drosophila Adult)", markerstrokewidth=0)

plot!(legend_plot, legend_columns=3, legendfontsize=8)


final_plot = plot(plots_array..., legend_plot, markersize=4,
                 layout=@layout([grid(2,4); c{0.15h}]),
                 size=(1000, 600), dpi=600, legendfontsize=10,
                 margin=3Plots.mm, plot_title="Performance by Task, normalised by Wiring Cost",xtickfontsize=10, titlefontsize=12, ylabelfontsize=12)

