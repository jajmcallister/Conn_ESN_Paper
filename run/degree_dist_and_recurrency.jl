

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
        in_degree = Base.count(x -> !iszero(x), W[:, node])   # Count nonzero entries in the column
        out_degree = Base.count(x -> !iszero(x), W[node, :]) # Count nonzero entries in the row
    return in_degree, out_degree
end

conn_esns = deepcopy(conn_ESNs)
er_esns   = deepcopy(er_ESNs)
cfg_esns  = deepcopy(cfg_ESNs)

conn_larva_degree_in  = Float64[]
conn_larva_degree_out = Float64[]
er_larva_degree_in    = Float64[]
er_larva_degree_out   = Float64[]
conn_adult_degree_in  = Float64[]
conn_adult_degree_out = Float64[]
er_adult_degree_in    = Float64[]
er_adult_degree_out   = Float64[]

for i in 3:11
    for j in 1:30
        N = size(conn_esns[i][j],1)
        for k in 1:N
            x, y  = node_degrees(conn_esns[i][j], k)
            z, zz = node_degrees(er_esns[i][j], k)
            push!(conn_larva_degree_in,  x/(N))
            push!(conn_larva_degree_out, y/(N))
            push!(er_larva_degree_in,    z/(N))
            push!(er_larva_degree_out,   zz/(N))
        end
    end
end

for i in 12:39
    for j in 1:30
        N = size(conn_esns[i][j],1)
        for k in 1:N
            x, y  = node_degrees(conn_esns[i][j], k)
            z, zz = node_degrees(er_esns[i][j], k)
            push!(conn_adult_degree_in,  x/(N))
            push!(conn_adult_degree_out, y/(N))
            push!(er_adult_degree_in,    z/(N))
            push!(er_adult_degree_out,   zz/(N))
        end
    end
end



# can visualise with density plots...
bw = 0.004
kd1 = kde(conn_larva_degree_in; bandwidth=bw)
kd2 = kde(conn_larva_degree_out; bandwidth=bw)
kd3 = kde(er_larva_degree_in; bandwidth=bw)
kd4 = kde(er_larva_degree_out; bandwidth=bw)
kd5 = kde(conn_adult_degree_in; bandwidth=bw)
kd6 = kde(conn_adult_degree_out; bandwidth=bw)
kd7 = kde(er_adult_degree_in; bandwidth=bw)
kd8 = kde(er_adult_degree_out; bandwidth=bw)

p1 = plot(kd1.x, kd1.density, lw=2, xlabel="Normalised degree", ylabel="Density", c=dodgerblues[2], label="Larva", title="In degree")
plot!(kd3.x, kd3.density, lw=2, xlabel="Normalised degree", ylabel="Density", c=crimsons[2], label="ER")
p2 = plot(kd2.x, kd2.density, lw=2, xlabel="Normalised degree", ylabel="Density", c=dodgerblues[2])
plot!(kd4.x, kd4.density, lw=2, xlabel="Normalised degree", ylabel="Density", c=crimsons[2], title="Out degree", legend=false)
p3 = plot(kd5.x, kd5.density, lw=2, xlabel="Normalised degree", ylabel="Density", c=dodgerblues[3])
plot!(kd7.x, kd7.density, lw=2, xlabel="Normalised degree", ylabel="Density", c=crimsons[3], legend=false)
p4 = plot(kd6.x, kd6.density, lw=2, xlabel="Normalised degree", ylabel="Density", c=dodgerblues[3], label="Adult")
plot!(kd8.x, kd8.density, lw=2, xlabel="Normalised degree", ylabel="Density", c=crimsons[3], label="ER")

plot(p1,p2,p3,p4,layout=(2,2),size=(800,800),xlim=(-0.02,0.125))

conn_larva_degree_in
default(fontfamily="Helvetica")
binss = 0:0.006:0.1
h1 = histogram(conn_larva_degree_in, bins=binss, normalize=true, label="Conn", lw=0, alpha=0.5, c=:dodgerblue4,ylabel="Density", title="Larva In-degree")
histogram!(er_larva_degree_in, bins=binss, normalize=true, label="ER", lw=0, alpha=0.5, c=:crimson)
h2 = histogram(conn_larva_degree_out, bins=binss, normalize=true, label="Larva", lw=0, alpha=0.5, c=:dodgerblue4, title="Larva Out-degree", legend=false)
histogram!(er_larva_degree_out, bins=binss, normalize=true, label="ER", lw=0, alpha=0.5, c=:crimson, legend=false)
h3 = histogram(conn_adult_degree_in, bins=binss, normalize=true, label="Adult", lw=0, alpha=0.7, c=dodgerblues[3], xlabel="Normalised in degree", ylabel="Density", title="Adult In-degree")
histogram!(er_adult_degree_in, bins=binss, normalize=true, label="ER", lw=0, alpha=0.5, c=crimsons[3],legend=false)
h4 = histogram(conn_adult_degree_out, bins=binss, normalize=true, label="Adult", lw=0, alpha=0.7, c=dodgerblues[3], xlabel="Normalised out degree", title="Adult Out-degree")
histogram!(er_adult_degree_out, bins=binss, normalize=true, label="ER", lw=0, alpha=0.5, c=crimsons[3], legend=false)


p = plot(h1,h2,h3,h4,layout=(2,2),size=(800,600), xlabelfontsize=14, legendfontsize=12, ylabelfontsize=14, xticks=0:0.05:0.1, xlim=(0,0.08), tickfontsize=18,ylim=(0,55),grid=false, plot_title="Normalised Degree Distributions")


using StatsBase, Plots

gcdf1 = StatsBase.ecdf(conn_larva_degree_in)
gcdf2 = StatsBase.ecdf(er_larva_degree_in)
h1 = plot(x -> gcdf1(x), -0.0, 0.125,
          label="Conn", c=:dodgerblue4,
          ylabel="CDF", title="Larva In-degree")
plot!(x -> gcdf2(x), -0.0, 0.125,
      label="ER", c=:crimson)
gcdf1 = StatsBase.ecdf(conn_larva_degree_out)
gcdf2 = StatsBase.ecdf(er_larva_degree_out)
h2 = plot(x -> gcdf1(x), -0.02, 0.125,
          label="Larva", c=:dodgerblue4,
          title="Larva Out-degree", legend=false)
plot!(x -> gcdf2(x), -0.02, 0.125,
      label="ER", c=:crimson, legend=false)
gcdf1 = StatsBase.ecdf(conn_adult_degree_in)
gcdf2 = StatsBase.ecdf(er_adult_degree_in)
h3 = plot(x -> gcdf1(x), -0.02, 0.125,
          label="Adult", c=dodgerblues[3],
          xlabel="Normalised in degree", ylabel="CDF",
          title="Adult In-degree")
plot!(x -> gcdf2(x), -0.02, 0.125,
      label="ER", c=crimsons[3], legend=false)
gcdf1 = StatsBase.ecdf(conn_adult_degree_out)
gcdf2 = StatsBase.ecdf(er_adult_degree_out)
h4 = plot(x -> gcdf1(x), -0.02, 0.125,
          label="Adult", c=dodgerblues[3],
          xlabel="Normalised out degree",
          title="Adult Out-degree")
plot!(x -> gcdf2(x), -0.02, 0.125,
      label="ER", c=crimsons[3], legend=false)



plot(h1,h2,h3,h4,
     layout=(2,2), size=(800,800),
     xlim=(-0.0,0.1), ylim=(0,1),
     xticks=0:0.05:0.1,
     xlabelfontsize=14, ylabelfontsize=14,
     legendfontsize=12, xtickfontsize=10, ytickfontsize=12,
     grid=false, plot_title="Cumulative Degree Distributions")


# Scatter plot of selfrecurrency vs. WTV
function get_recurrent(mat)
    sz = size(mat, 1)
    recs = []
    for i in 1:sz
        if mat[i,i]!=0.0
            push!(recs,1)
        else
            push!(recs,0)
        end
    end
    return recs
end

lenn = length(weighted_tv_conn[1][1])
recs_conn = [[get_recurrent(Matrix(conn_ESNs[i][j])) for j in 1:lenn] for i in 1:39]
recs_er = [[get_recurrent(Matrix(er_ESNs[i][j])) for j in 1:lenn] for i in 1:39]
recs_cfg = [[get_recurrent(Matrix(cfg_ESNs[i][j])) for j in 1:lenn] for i in 1:39]

larva_recs_conn = [mean(mean.(recs_conn[i])) for i in 3:11]
adult_recs_conn = [mean(mean.(recs_conn[i])) for i in 12:39]
larva_recs_er   = [mean(mean.(recs_er[i])) for i in 3:11]
adult_recs_er   = [mean(mean.(recs_er[i])) for i in 12:39]
larva_recs_cfg  = [mean(mean.(recs_cfg[i])) for i in 3:11]
adult_recs_cfg  = [mean(mean.(recs_cfg[i])) for i in 12:39]

s = bar([1],[mean(larva_recs_conn)], c=:dodgerblue4, label="Conn", yerror=std(larva_recs_conn)/sqrt(9), alpha=0.8)
bar!([2],[mean(larva_recs_er)], c=:crimson, label="ER", yerror=std(larva_recs_er)/sqrt(9), alpha=0.8)
bar!([3],[mean(larva_recs_cfg)], c=:orange, label="CFG", yerror=std(larva_recs_cfg)/sqrt(9), alpha=0.8)

bar!([5],[mean(adult_recs_conn)], c=dodgerblues[3], label=false, yerror=std(adult_recs_conn)/sqrt(28))
bar!([6],[mean(adult_recs_er)], c=crimsons[3], label=false, yerror=std(adult_recs_er)/sqrt(28))
bar!([7],[mean(adult_recs_cfg)], c=:darkorange, label=false, yerror=std(adult_recs_cfg)/sqrt(28))

using Plots.PlotMeasures
plot!(ylabel="\n Mean Self Recurrency \n", xticks=([2,6],["Larva", "Adult"]), title="Self Recurrency", ylim=(0,0.2), yticks=0:0.1:0.2, size=(500,300), grid=false)
plot!(ylabelfontsize=14, ytickfontsize=12, titlefontsize=16, legendfontsize=10, xtickfontsize=12, leftmargin=5mm)

default(fontfamily="Helvetica")


# Compute correlations for each subnetwork and each task
correlations_conn = [[round(cor(vcat(recs_conn[i]...), vcat(weighted_tv_conn[i][t]...)), digits=3) for t in 1:8] for i in 3:39]
correlations_er   = [[round(cor(vcat(recs_er[i]...), vcat(weighted_tv_rand[i][t]...)), digits=3) for t in 1:8] for i in 3:39]
correlations_cfg  = [[round(cor(vcat(recs_cfg[i]...), vcat(weighted_tv_cfg[i][t]...)), digits=3) for t in 1:8] for i in 3:39]

m11 = ([correlations_conn[i][1] for i in 1:37])
std11 = std([correlations_conn[i][1] for i in 1:37])
m12 = ([correlations_er[i][1] for i in 1:37])
std12 = std([correlations_er[i][1] for i in 1:37])
m13 = ([correlations_cfg[i][1] for i in 1:37])
std13 = std([correlations_cfg[i][1] for i in 1:37])

m21 = ([correlations_conn[i][2] for i in 1:37])
std21 = std([correlations_conn[i][2] for i in 1:37])
m22 = ([correlations_er[i][2] for i in 1:37])
std22 = std([correlations_er[i][2] for i in 1:37])
m23 = ([correlations_cfg[i][2] for i in 1:37])
std23 = std([correlations_cfg[i][2] for i in 1:37])

m31 = ([correlations_conn[i][3] for i in 1:37])
std31 = std([correlations_conn[i][3] for i in 1:37])
m32 = ([correlations_er[i][3] for i in 1:37])
std32 = std([correlations_er[i][3] for i in 1:37])
m33 = ([correlations_cfg[i][3] for i in 1:37])
std33 = std([correlations_cfg[i][3] for i in 1:37])

m41 = ([correlations_conn[i][4] for i in 1:37])
std41 = std([correlations_conn[i][4] for i in 1:37])
m42 = ([correlations_er[i][4] for i in 1:37])
std42 = std([correlations_er[i][4] for i in 1:37])
m43 = ([correlations_cfg[i][4] for i in 1:37])
std43 = std([correlations_cfg[i][4] for i in 1:37])

m51 = ([correlations_conn[i][5] for i in 1:37])
std51 = std([correlations_conn[i][5] for i in 1:37])
m52 = ([correlations_er[i][5] for i in 1:37])
std52 = std([correlations_er[i][5] for i in 1:37])
m53 = ([correlations_cfg[i][5] for i in 1:37])
std53 = std([correlations_cfg[i][5] for i in 1:37])

m61 = ([correlations_conn[i][6] for i in 1:37])
std61 = std([correlations_conn[i][6] for i in 1:37])
m62 = ([correlations_er[i][6] for i in 1:37])
std62 = std([correlations_er[i][6] for i in 1:37])
m63 = ([correlations_cfg[i][6] for i in 1:37])
std63 = std([correlations_cfg[i][6] for i in 1:37])

m71 = ([correlations_conn[i][7] for i in 1:37])
std71 = std([correlations_conn[i][7] for i in 1:37])
m72 = ([correlations_er[i][7] for i in 1:37])
std72 = std([correlations_er[i][7] for i in 1:37])
m73 = ([correlations_cfg[i][7] for i in 1:37])
std73 = std([correlations_cfg[i][7] for i in 1:37])

m81 = ([correlations_conn[i][8] for i in 1:37])
std81 = std([correlations_conn[i][8] for i in 1:37])
m82 = ([correlations_er[i][8] for i in 1:37])
std82 = std([correlations_er[i][8] for i in 1:37])
m83 = ([correlations_cfg[i][8] for i in 1:37])
std83 = std([correlations_cfg[i][8] for i in 1:37])

m_conns = [m11, m21, m31, m41, m51, m61, m71, m81]
m_ers = [m12, m22, m32, m42, m52, m62, m72, m82]
m_cfgs = [m13, m23, m33, m43, m53, m63, m73, m83]

jitter(x_base, n_points; width=0.3) = x_base .+ (rand(n_points) * 2 * width .- width)

# Create the initial plot canvas
# By creating the plot first, all subsequent calls can use scatter!
p1 = plot()

# Define the base x-positions and the number of data columns to plot
x_bases = [1, 5, 9, 13, 17, 21, 25, 29] # Base x for 'conn' data for each task
num_tasks = 8

# Loop through each task/column of data
for i in 1:num_tasks
    col = i
    x_base = x_bases[i]

    # # Group 3 (rows 12:end)
    n_end = length(12:39) # Safely get number of points
    scatter!(p1, jitter(x_base, n_end), m_conns[i][10:end], c=dodgerblues[3], markerstrokewidth=0, label=false)
    scatter!(p1, jitter(x_base + 1, n_end), m_ers[i][10:end], c=crimsons[3], markerstrokewidth=0, label=false)
    # scatter!(p1, jitter(x_base + 2, n_end), m_cfgs[i][10:end], c=oranges[3], markerstrokewidth=0, label=false)

    
    # Group 2 (rows 3:11)
    scatter!(p1, jitter(x_base, 9), m_conns[i][1:9], c=:dodgerblue4, markerstrokewidth=0, label=false)
    scatter!(p1, jitter(x_base + 1, 9), m_ers[i][1:9], c=:crimson, markerstrokewidth=0, label=false)
    # scatter!(p1, jitter(x_base + 2, 9), m_cfgs[i][1:9], c=:orange, markerstrokewidth=0, label=false)

end

# Add vertical lines and final plot settings
vline!(p1, [4, 8, 12, 16, 20, 24, 28] .- 0.5, c=:black, linestyle=:dash, label=false, grid=false)
hline!(p1, [0.0], c=:grey, lw=3, label=false)
plot!(p1, xticks=([2, 6, 10, 14, 18, 22, 26, 30] .- 0.5, ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8"]), 
title="WTV and Self-Recurrency Correlation", ylim=(-0.5,0.5))


# Scatter plot of correlations between node degree and WTV
#OUT DEGREE
recs_conn_out = [[node_degrees_total(Matrix(conn_ESNs[z][j])).out_degrees for j in 1:lenn] for z in 1:39] 
recs_er_out = [[node_degrees_total(Matrix(er_ESNs[z][j])).out_degrees for j in 1:lenn] for z in 1:39]
recs_cfg_out = [[node_degrees_total(Matrix(cfg_ESNs[z][j])).out_degrees for j in 1:lenn] for z in 1:39]


using LaTeXStrings

# Compute correlations for each subnetwork and each task
correlations_conn = [[round(mean([cor(recs_conn_out[i][z],weighted_tv_conn[i][t][z]) for z in 1:lenn]), digits=8) for t in 1:8] for i in 1:39]
correlations_er   = [[round(mean([cor(recs_er_out[i][z],weighted_tv_rand[i][t][z]) for z in 1:lenn]), digits=8) for t in 1:8] for i in 1:39]
correlations_cfg  = [[round(mean([cor(recs_cfg_out[i][z],weighted_tv_cfg[i][t][z]) for z in 1:lenn]), digits=8) for t in 1:8] for i in 1:39]


m11 = ([correlations_conn[i][1] for i in 1:39])
std11 = std([correlations_conn[i][1] for i in 1:39])
m12 = ([correlations_er[i][1] for i in 1:39])
std12 = std([correlations_er[i][1] for i in 1:39])
m13 = ([correlations_cfg[i][1] for i in 1:39])
std13 = std([correlations_cfg[i][1] for i in 1:39])

m21 = ([correlations_conn[i][2] for i in 1:39])
std21 = std([correlations_conn[i][2] for i in 1:39])
m22 = ([correlations_er[i][2] for i in 1:39])
std22 = std([correlations_er[i][2] for i in 1:39])
m23 = ([correlations_cfg[i][2] for i in 1:39])
std23 = std([correlations_cfg[i][2] for i in 1:39])

m31 = ([correlations_conn[i][3] for i in 1:39])
std31 = std([correlations_conn[i][3] for i in 1:39])
m32 = ([correlations_er[i][3] for i in 1:39])
std32 = std([correlations_er[i][3] for i in 1:39])
m33 = ([correlations_cfg[i][3] for i in 1:39])
std33 = std([correlations_cfg[i][3] for i in 1:39])

m41 = ([correlations_conn[i][4] for i in 1:39])
std41 = std([correlations_conn[i][4] for i in 1:39])
m42 = ([correlations_er[i][4] for i in 1:39])
std42 = std([correlations_er[i][4] for i in 1:39])
m43 = ([correlations_cfg[i][4] for i in 1:39])
std43 = std([correlations_cfg[i][4] for i in 1:39])

m51 = ([correlations_conn[i][5] for i in 1:39])
std51 = std([correlations_conn[i][5] for i in 1:39])
m52 = ([correlations_er[i][5] for i in 1:39])
std52 = std([correlations_er[i][5] for i in 1:39])
m53 = ([correlations_cfg[i][5] for i in 1:39])
std53 = std([correlations_cfg[i][5] for i in 1:39])

m61 = ([correlations_conn[i][6] for i in 1:39])
std61 = std([correlations_conn[i][6] for i in 1:39])
m62 = ([correlations_er[i][6] for i in 1:39])
std62 = std([correlations_er[i][6] for i in 1:39])
m63 = ([correlations_cfg[i][6] for i in 1:39])
std63 = std([correlations_cfg[i][6] for i in 1:39])

m71 = ([correlations_conn[i][7] for i in 1:39])
std71 = std([correlations_conn[i][7] for i in 1:39])
m72 = ([correlations_er[i][7] for i in 1:39])
std72 = std([correlations_er[i][7] for i in 1:39])
m73 = ([correlations_cfg[i][7] for i in 1:39])
std73 = std([correlations_cfg[i][7] for i in 1:39])

m81 = ([correlations_conn[i][8] for i in 1:39])
std81 = std([correlations_conn[i][8] for i in 1:39])
m82 = ([correlations_er[i][8] for i in 1:39])
std82 = std([correlations_er[i][8] for i in 1:39])
m83 = ([correlations_cfg[i][8] for i in 1:39])
std83 = std([correlations_cfg[i][8] for i in 1:39])

m_conns = [m11, m21, m31, m41, m51, m61, m71, m81]
m_ers = [m12, m22, m32, m42, m52, m62, m72, m82]
m_cfgs = [m13, m23, m33, m43, m53, m63, m73, m83]


p2 = plot()

# Define the base x-positions and the number of data columns to plot
x_bases = [1, 5, 9, 13, 17, 21, 25, 29] # Base x for 'conn' data for each task
num_tasks = 8
m_conns[1]

# Loop through each task/column of data
for i in 1:num_tasks
    col = i
    x_base = x_bases[i]

    # # Group 3 (rows 12:end)
    n_end = length(12:39) # Safely get number of points
    scatter!(p2, jitter(x_base, n_end), m_conns[i][12:end], c=dodgerblues[3], markerstrokewidth=0, label=false)
    scatter!(p2, jitter(x_base + 1, n_end), m_ers[i][12:end], c=crimsons[3], markerstrokewidth=0, label=false)
    scatter!(p2, jitter(x_base + 2, n_end), m_cfgs[i][12:end], c=oranges[3], markerstrokewidth=0, label=false)

    
    # Group 2 (rows 3:11)
    scatter!(p2, jitter(x_base, 9), m_conns[i][3:11], c=:dodgerblue4, markerstrokewidth=0, label=false)
    scatter!(p2, jitter(x_base + 1, 9), m_ers[i][3:11], c=:crimson, markerstrokewidth=0, label=false)
    scatter!(p2, jitter(x_base + 2, 9), m_cfgs[i][3:11], c=:orange, markerstrokewidth=0, label=false)

end

# Add vertical lines and final plot settings
vline!(p2, [4, 8, 12, 16, 20, 24, 28], c=:black, linestyle=:dash, label=false, grid=false)
hline!(p2, [0.0], c=:grey, lw=3, label=false)
plot!(p2, xticks=([2, 6, 10, 14, 18, 22, 26, 30], ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8"]), 
title="WTV and Node Out-Degree Correlation", ylim=(-1,1))




#IN DEGREE
recs_conn_in = [[node_degrees_total(Matrix(conn_ESNs[z][j])).in_degrees for j in 1:lenn] for z in 1:39] 
recs_er_in = [[node_degrees_total(Matrix(er_ESNs[z][j])).in_degrees for j in 1:lenn] for z in 1:39]
recs_cfg_in = [[node_degrees_total(Matrix(cfg_ESNs[z][j])).in_degrees for j in 1:lenn] for z in 1:39]

# Compute correlations for each subnetwork and each task
correlations_conn = [[round(mean([cor(recs_conn_in[i][z],weighted_tv_conn[i][t][z]) for z in 1:lenn]), digits=8) for t in 1:8] for i in 1:39]
correlations_er   = [[round(mean([cor(recs_er_in[i][z],weighted_tv_rand[i][t][z]) for z in 1:lenn]), digits=8) for t in 1:8] for i in 1:39]
correlations_cfg  = [[round(mean([cor(recs_cfg_in[i][z],weighted_tv_cfg[i][t][z]) for z in 1:lenn]), digits=8) for t in 1:8] for i in 1:39]


m11 = ([correlations_conn[i][1] for i in 1:39])
std11 = std([correlations_conn[i][1] for i in 1:39])
m12 = ([correlations_er[i][1] for i in 1:39])
std12 = std([correlations_er[i][1] for i in 1:39])
m13 = ([correlations_cfg[i][1] for i in 1:39])
std13 = std([correlations_cfg[i][1] for i in 1:39])

m21 = ([correlations_conn[i][2] for i in 1:39])
std21 = std([correlations_conn[i][2] for i in 1:39])
m22 = ([correlations_er[i][2] for i in 1:39])
std22 = std([correlations_er[i][2] for i in 1:39])
m23 = ([correlations_cfg[i][2] for i in 1:39])
std23 = std([correlations_cfg[i][2] for i in 1:39])

m31 = ([correlations_conn[i][3] for i in 1:39])
std31 = std([correlations_conn[i][3] for i in 1:39])
m32 = ([correlations_er[i][3] for i in 1:39])
std32 = std([correlations_er[i][3] for i in 1:39])
m33 = ([correlations_cfg[i][3] for i in 1:39])
std33 = std([correlations_cfg[i][3] for i in 1:39])

m41 = ([correlations_conn[i][4] for i in 1:39])
std41 = std([correlations_conn[i][4] for i in 1:39])
m42 = ([correlations_er[i][4] for i in 1:39])
std42 = std([correlations_er[i][4] for i in 1:39])
m43 = ([correlations_cfg[i][4] for i in 1:39])
std43 = std([correlations_cfg[i][4] for i in 1:39])

m51 = ([correlations_conn[i][5] for i in 1:39])
std51 = std([correlations_conn[i][5] for i in 1:39])
m52 = ([correlations_er[i][5] for i in 1:39])
std52 = std([correlations_er[i][5] for i in 1:39])
m53 = ([correlations_cfg[i][5] for i in 1:39])
std53 = std([correlations_cfg[i][5] for i in 1:39])

m61 = ([correlations_conn[i][6] for i in 1:39])
std61 = std([correlations_conn[i][6] for i in 1:39])
m62 = ([correlations_er[i][6] for i in 1:39])
std62 = std([correlations_er[i][6] for i in 1:39])
m63 = ([correlations_cfg[i][6] for i in 1:39])
std63 = std([correlations_cfg[i][6] for i in 1:39])

m71 = ([correlations_conn[i][7] for i in 1:39])
std71 = std([correlations_conn[i][7] for i in 1:39])
m72 = ([correlations_er[i][7] for i in 1:39])
std72 = std([correlations_er[i][7] for i in 1:39])
m73 = ([correlations_cfg[i][7] for i in 1:39])
std73 = std([correlations_cfg[i][7] for i in 1:39])

m81 = ([correlations_conn[i][8] for i in 1:39])
std81 = std([correlations_conn[i][8] for i in 1:39])
m82 = ([correlations_er[i][8] for i in 1:39])
std82 = std([correlations_er[i][8] for i in 1:39])
m83 = ([correlations_cfg[i][8] for i in 1:39])
std83 = std([correlations_cfg[i][8] for i in 1:39])

m_conns = [m11, m21, m31, m41, m51, m61, m71, m81]
m_ers = [m12, m22, m32, m42, m52, m62, m72, m82]
m_cfgs = [m13, m23, m33, m43, m53, m63, m73, m83]


p3 = plot()

# Define the base x-positions and the number of data columns to plot
x_bases = [1, 5, 9, 13, 17, 21, 25, 29] # Base x for 'conn' data for each task
num_tasks = 8
m_conns[1]

# Loop through each task/column of data
for i in 1:num_tasks
    col = i
    x_base = x_bases[i]

    # # Group 3 (rows 12:end)
    n_end = length(12:39) # Safely get number of points
    scatter!(p3, jitter(x_base, n_end), m_conns[i][12:end], c=dodgerblues[3], markerstrokewidth=0, label=false)
    scatter!(p3, jitter(x_base + 1, n_end), m_ers[i][12:end], c=crimsons[3], markerstrokewidth=0, label=false)
    scatter!(p3, jitter(x_base + 2, n_end), m_cfgs[i][12:end], c=oranges[3], markerstrokewidth=0, label=false)

    
    # Group 2 (rows 3:11)
    scatter!(p3, jitter(x_base, 9), m_conns[i][3:11], c=:dodgerblue4, markerstrokewidth=0, label=false)
    scatter!(p3, jitter(x_base + 1, 9), m_ers[i][3:11], c=:crimson, markerstrokewidth=0, label=false)
    scatter!(p3, jitter(x_base + 2, 9), m_cfgs[i][3:11], c=:orange, markerstrokewidth=0, label=false)

end

# Add vertical lines and final plot settings
vline!(p3, [4, 8, 12, 16, 20, 24, 28], c=:black, linestyle=:dash, label=false, grid=false)
hline!(p3, [0.0], c=:grey, lw=3, label=false)
plot!(p3, xticks=([2, 6, 10, 14, 18, 22, 26, 30], ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8"]), 
title="WTV and Node In-Degree Correlation", ylim=(-0.5,0.5))




using Plots.PlotMeasures
pp4 = plot(p1,p3,p2,layout=(3,1),size=(700,900),leftmargin=10mm,dpi=400)

plot(p,pp4,layout=(1,2),size=(1600,800),leftmargin=10mm,dpi=400)






function node_degrees_total(W::AbstractMatrix)
    # Treat any nonzero entry as an edge
    in_degrees  = vec(sum(W .!= 0, dims=1))  # columns: incoming edges
    out_degrees = vec(sum(W .!= 0, dims=2))  # rows: outgoing edges
    total_degrees = in_degrees + out_degrees
    return (in_degrees=in_degrees, out_degrees=out_degrees, total_degrees=total_degrees)
end


lenn=30
conn_degrees = [[node_degrees_total(Matrix(conn_ESNs[z][j])).total_degrees for j in 1:lenn] for z in 1:39] 
er_degrees = [[node_degrees_total(Matrix(er_ESNs[z][j])).total_degrees for j in 1:lenn] for z in 1:39]
cfg_degrees = [[node_degrees_total(Matrix(cfg_ESNs[z][j])).total_degrees for j in 1:lenn] for z in 1:39]


# Compute correlations for each subnetwork and each task
correlations_conn = [[round(mean([cor(conn_degrees[i][z],weighted_tv_conn[i][t][z]) for z in 1:lenn]), digits=8) for t in 1:8] for i in 3:39]
correlations_er   = [[round(mean([cor(er_degrees[i][z],weighted_tv_rand[i][t][z]) for z in 1:lenn]), digits=8) for t in 1:8] for i in 3:39]
correlations_cfg  = [[round(mean([cor(cfg_degrees[i][z],weighted_tv_cfg[i][t][z]) for z in 1:lenn]), digits=8) for t in 1:8] for i in 3:39]


m11 = ([correlations_conn[i][1] for i in 1:37])
std11 = std([correlations_conn[i][1] for i in 1:37])
m12 = ([correlations_er[i][1] for i in 1:37])
std12 = std([correlations_er[i][1] for i in 1:37])
m13 = ([correlations_cfg[i][1] for i in 1:37])
std13 = std([correlations_cfg[i][1] for i in 1:37])

m21 = ([correlations_conn[i][2] for i in 1:37])
std21 = std([correlations_conn[i][2] for i in 1:37])
m22 = ([correlations_er[i][2] for i in 1:37])
std22 = std([correlations_er[i][2] for i in 1:37])
m23 = ([correlations_cfg[i][2] for i in 1:37])
std23 = std([correlations_cfg[i][2] for i in 1:37])

m31 = ([correlations_conn[i][3] for i in 1:37])
std31 = std([correlations_conn[i][3] for i in 1:37])
m32 = ([correlations_er[i][3] for i in 1:37])
std32 = std([correlations_er[i][3] for i in 1:37])
m33 = ([correlations_cfg[i][3] for i in 1:37])
std33 = std([correlations_cfg[i][3] for i in 1:37])

m41 = ([correlations_conn[i][4] for i in 1:37])
std41 = std([correlations_conn[i][4] for i in 1:37])
m42 = ([correlations_er[i][4] for i in 1:37])
std42 = std([correlations_er[i][4] for i in 1:37])
m43 = ([correlations_cfg[i][4] for i in 1:37])
std43 = std([correlations_cfg[i][4] for i in 1:37])

m51 = ([correlations_conn[i][5] for i in 1:37])
std51 = std([correlations_conn[i][5] for i in 1:37])
m52 = ([correlations_er[i][5] for i in 1:37])
std52 = std([correlations_er[i][5] for i in 1:37])
m53 = ([correlations_cfg[i][5] for i in 1:37])
std53 = std([correlations_cfg[i][5] for i in 1:37])

m61 = ([correlations_conn[i][6] for i in 1:37])
std61 = std([correlations_conn[i][6] for i in 1:37])
m62 = ([correlations_er[i][6] for i in 1:37])
std62 = std([correlations_er[i][6] for i in 1:37])
m63 = ([correlations_cfg[i][6] for i in 1:37])
std63 = std([correlations_cfg[i][6] for i in 1:37])

m71 = ([correlations_conn[i][7] for i in 1:37])
std71 = std([correlations_conn[i][7] for i in 1:37])
m72 = ([correlations_er[i][7] for i in 1:37])
std72 = std([correlations_er[i][7] for i in 1:37])
m73 = ([correlations_cfg[i][7] for i in 1:37])
std73 = std([correlations_cfg[i][7] for i in 1:37])

m81 = ([correlations_conn[i][8] for i in 1:37])
std81 = std([correlations_conn[i][8] for i in 1:37])
m82 = ([correlations_er[i][8] for i in 1:37])
std82 = std([correlations_er[i][8] for i in 1:37])
m83 = ([correlations_cfg[i][8] for i in 1:37])
std83 = std([correlations_cfg[i][8] for i in 1:37])

m_conns = [m11, m21, m31, m41, m51, m61, m71, m81]
m_ers = [m12, m22, m32, m42, m52, m62, m72, m82]
m_cfgs = [m13, m23, m33, m43, m53, m63, m73, m83]


p66 = plot()

# Define the base x-positions and the number of data columns to plot
x_bases = [1, 5, 9, 13, 17, 21, 25, 29] # Base x for 'conn' data for each task
num_tasks = 8
m_conns[1]

# Loop through each task/column of data
for i in 1:num_tasks
    col = i
    x_base = x_bases[i]

    # # Group 3 (rows 12:end)
    n_end = length(12:39) # Safely get number of points
    scatter!(p66, jitter(x_base, n_end), m_conns[i][10:end], c=dodgerblues[3], markerstrokewidth=0, label=false)
    scatter!(p66, jitter(x_base + 1, n_end), m_ers[i][10:end], c=crimsons[3], markerstrokewidth=0, label=false)
    # scatter!(p66, jitter(x_base + 2, n_end), m_cfgs[i][10:end], c=oranges[3], markerstrokewidth=0, label=false)

    
    # Group 2 (rows 3:11)
    scatter!(p66, jitter(x_base, 9), m_conns[i][1:9], c=:dodgerblue4, markerstrokewidth=0, label=false)
    scatter!(p66, jitter(x_base + 1, 9), m_ers[i][1:9], c=:crimson, markerstrokewidth=0, label=false)
    # scatter!(p66, jitter(x_base + 2, 9), m_cfgs[i][1:9], c=:orange, markerstrokewidth=0, label=false)

end

# Add vertical lines and final plot settings
vline!(p66, [4, 8, 12, 16, 20, 24, 28] .- 0.5, c=:black, linestyle=:dash, label=false, grid=false)
hline!(p66, [0.0], c=:grey, lw=3, label=false)
plot!(p66, xticks=([2, 6, 10, 14, 18, 22, 26, 30] .- 0.5, ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8"]), 
title="WTV and Node-Degree Correlation", ylim=(-1,1), labelfontsize=20)

default(fontfamily="Helvetica")
ppp = plot(p66,p4,p1, layout=(3,1),size=(700,900),leftmargin=10mm,ylim=(-0.35,0.65), dpi=400, tickfontsize=18)
###############


p_four = plot(p66, p4, p1, p_betweenness, layout=(2,2), size=(800,600), xlabelfontsize=18, leftmargin=10mm, dpi=400)


########################################
########################################
# plot example scatterplot

weighted_tv_conn[3]
for iddd in 3:11
    println("Subnetwork $iddd:")
    println("Conn ", cor(conn_degrees[iddd][1], weighted_tv_conn[iddd][7][1]))
    println("Rand ", cor(er_degrees[iddd][1], weighted_tv_rand[iddd][7][1]))
end

idd=4
taskid=1

# Extract x (degrees) and y (weighted task variance)
x_conn = conn_degrees[idd][1]
y_conn = weighted_tv_conn[idd][taskid][1]

x_rand = er_degrees[idd][1]
y_rand = weighted_tv_rand[idd][taskid][1]

# --- correlation coefficients ---
corr_conn = cor(x_conn, y_conn)
corr_rand = cor(x_rand, y_rand)

# --- best-fit line (linear regression) ---
# slope and intercept via least squares
m_conn = cov(x_conn, y_conn) / var(x_conn)
b_conn = mean(y_conn) - m_conn * mean(x_conn)

m_rand = cov(x_rand, y_rand) / var(x_rand)
b_rand = mean(y_rand) - m_rand * mean(x_rand)

# x-range for plotting the regression line
x_line = range(minimum([minimum(x_conn), minimum(x_rand)]),
               maximum([maximum(x_conn), maximum(x_rand)]),
               length=100)

# y-values
y_line_conn = m_conn .* x_line .+ b_conn
y_line_rand = m_rand .* x_line .+ b_rand

# scatter + lines
s11 = scatter(
    x_conn, y_conn,
    grid=false, c=dodgerblues[1], label=" Conn",
    markerstrokewidth=0
)
plot!(s11, x_line, y_line_conn, lw=3, c=dodgerblues[2],
      label=" Corr = $(round(corr_conn, digits=2))")
scatter!(
    x_conn, y_conn,
    grid=false, c=dodgerblues[1], label=false,
    markerstrokewidth=0
)

s12 = scatter(
    x_rand, y_rand,
    grid=false, c=crimsons[1], label=" Random",
    markerstrokewidth=0
)
plot!(s12, x_line, y_line_rand, lw=3, c=crimsons[2],
      label=" Corr = $(round(corr_rand, digits=2))")
scatter!(
    x_rand, y_rand,
    grid=false, c=crimsons[1], label=false,
    markerstrokewidth=0
)

# combine
p11 = plot(s11, s12,
           layout=(1,2),
           legendfontsize=14,
           tickfontsize=14,
           size=(800,400),
           ylim=(-0.002,0.05),
           xlim=(-0.18,18),
           dpi=400,
           plot_title="Task 1: Working Memory")




idd=4
taskid=7

# Extract x (degrees) and y (weighted task variance)
x_conn = conn_degrees[idd][1]
y_conn = weighted_tv_conn[idd][taskid][1]

x_rand = er_degrees[idd][1]
y_rand = weighted_tv_rand[idd][taskid][1]

# --- correlation coefficients ---
corr_conn = cor(x_conn, y_conn)
corr_rand = cor(x_rand, y_rand)

# --- best-fit line (linear regression) ---
# slope and intercept via least squares
m_conn = cov(x_conn, y_conn) / var(x_conn)
b_conn = mean(y_conn) - m_conn * mean(x_conn)

m_rand = cov(x_rand, y_rand) / var(x_rand)
b_rand = mean(y_rand) - m_rand * mean(x_rand)

# x-range for plotting the regression line
x_line = range(minimum([minimum(x_conn), minimum(x_rand)]),
               maximum([maximum(x_conn), maximum(x_rand)]),
               length=100)

# y-values
y_line_conn = m_conn .* x_line .+ b_conn
y_line_rand = m_rand .* x_line .+ b_rand

# scatter + lines
s111 = scatter(
    x_conn, y_conn,
    grid=false, c=dodgerblues[1], label=" Conn",
    markerstrokewidth=0
)
plot!(s111, x_line, y_line_conn, lw=3, c=dodgerblues[2],
      label=" Corr = $(round(corr_conn, digits=2))")
scatter!(
    x_conn, y_conn,
    grid=false, c=dodgerblues[1], label=false,
    markerstrokewidth=0)

s121 = scatter(
    x_rand, y_rand,
    grid=false, c=crimsons[1], label=" Random",
    markerstrokewidth=0
)
plot!(s121, x_line, y_line_rand, lw=3, c=crimsons[2],
      label=" Corr = $(round(corr_rand, digits=2))")
scatter!(
    x_rand, y_rand,
    grid=false, c=crimsons[1], label=false,
    markerstrokewidth=0
)

# combine
p111 = plot(s111, s121,
           layout=(1,2),
           legendfontsize=14,
           tickfontsize=14,
           size=(800,400),
           ylim=(-0.006,0.2),
           xlim=(-0.18,18),
           dpi=400,
           plot_title="Task 7: Lorenz Prediction")

p = plot(p11, p111, layout=(2,1), size=(700,900), tickfontsize=18, leftmargin=5mm)

savefig(p, "C://Users/B00955735/OneDrive - Ulster University/Desktop/scatter_examples.svg")





#########################
# self-recurrency

for iddd in 3:11
    println("Subnetwork $iddd:")
    println("Conn ", cor(recs_conn[iddd][1], weighted_tv_conn[iddd][1][1]))
    println("Rand ", cor(recs_er[iddd][1], weighted_tv_rand[iddd][1][1]))
end

idd=7
taskid=1
# Extract x (degrees) and y (weighted task variance)
x_conn = recs_conn[idd][1]
y_conn = weighted_tv_conn[idd][taskid][1]

x_rand = recs_er[idd][1]
y_rand = weighted_tv_rand[idd][taskid][1]

# correlation coefficients
corr_conn = cor(x_conn, y_conn)
corr_rand = cor(x_rand, y_rand)

# best-fit line (linear regression)
# slope and intercept via least squares
m_conn = cov(x_conn, y_conn) / var(x_conn)
b_conn = mean(y_conn) - m_conn * mean(x_conn)

m_rand = cov(x_rand, y_rand) / var(x_rand)
b_rand = mean(y_rand) - m_rand * mean(x_rand)

# x-range for plotting the regression line
x_line = range(minimum([minimum(x_conn), minimum(x_rand)]),
               maximum([maximum(x_conn), maximum(x_rand)]),
               length=100)

# y-values
y_line_conn = m_conn .* x_line .+ b_conn
y_line_rand = m_rand .* x_line .+ b_rand

# scatter + lines
s31 = scatter(
    x_conn, y_conn,
    grid=false, c=dodgerblues[1], label=" Conn",
    markerstrokewidth=0
)
plot!(x_line, y_line_conn, lw=3, c=dodgerblues[2],
      label=" Corr = $(round(corr_conn, digits=2))")

s32 = scatter(
    x_rand, y_rand,
    grid=false, c=crimsons[1], label=" Random",
    markerstrokewidth=0
)
plot!(x_line, y_line_rand, lw=3, c=crimsons[2],
      label=" Corr = $(round(corr_rand, digits=2))")

# combine
p31 = plot(s31, s32,
           layout=(1,2),
           legendfontsize=14,
           tickfontsize=14,
           size=(800,400),
           leftmargin=10mm,
           dpi=400,
           plot_title="Task 1: Working Memory")


##############
# local clustering coefficient


for iddd in 3:11
    println("Subnetwork $iddd:")
    println("Conn ", cor(asps_conn[iddd][1], weighted_tv_conn[iddd][1][1]))
    println("Rand ", cor(asps_er[iddd][1], weighted_tv_rand[iddd][1][1]))
end

idd=9
taskid=1
# Extract x (degrees) and y (weighted task variance)
x_conn = asps_conn[idd][1]
y_conn = weighted_tv_conn[idd][taskid][1]

x_rand = asps_er[idd][1]
y_rand = weighted_tv_rand[idd][taskid][1]

#correlation coefficients
corr_conn = cor(x_conn, y_conn)
corr_rand = cor(x_rand, y_rand)
# best-fit line (linear regression)
# slope and intercept via least squares
m_conn = cov(x_conn, y_conn) / var(x_conn)
b_conn = mean(y_conn) - m_conn * mean(x_conn)
m_rand = cov(x_rand, y_rand) / var(x_rand)
b_rand = mean(y_rand) - m_rand * mean(x_rand)
# x-range for plotting the regression line
x_line = range(minimum([minimum(x_conn), minimum(x_rand)]),
               maximum([maximum(x_conn), maximum(x_rand)]),
               length=100)
# y-values
y_line_conn = m_conn .* x_line .+ b_conn
y_line_rand = m_rand .* x_line .+ b_rand
# scatter + lines
s21 = scatter(
    x_conn, y_conn,
    grid=false, c=dodgerblues[1], label=" Conn",
    markerstrokewidth=0
)
plot!(s21, x_line, y_line_conn, lw=3, c=dodgerblues[2],
      label=" Corr = $(round(corr_conn, digits=2))")
s22 = scatter(
    x_rand, y_rand,
    grid=false, c=crimsons[1], label=" Random",
    markerstrokewidth=0
)
plot!(s22, x_line, y_line_rand, lw=3, c=crimsons[2],
      label=" Corr = $(round(corr_rand, digits=2))")
# combine
p21 = plot(s21, s22,
           layout=(1,2),
           legendfontsize=14,
           tickfontsize=14,
           size=(800,400),
           leftmargin=10mm,
           dpi=400,
           plot_title="Task 1: Working Memory")


           plot(p11, p21, p31, layout=(3,1), size=(800,1200), leftmargin=10mm, dpi=400)