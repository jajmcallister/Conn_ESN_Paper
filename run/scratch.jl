
using LinearAlgebra

conn_ESNs_baseline = [[ConnectomeFunctions.randomise_nonzero_elements(deepcopy(conn_ESNs[i][j])) for j in 1:30] for i in 1:39]
er_ESNs_baseline = [[ConnectomeFunctions.randomise_nonzero_elements(deepcopy(er_ESNs[i][j])) for j in 1:30] for i in 1:39]
cfg_ESNs_baseline = [[ConnectomeFunctions.randomise_nonzero_elements(deepcopy(cfg_ESNs[i][j])) for j in 1:30] for i in 1:39]

conn_srs_baseline = [[maximum(abs.(eigvals(conn_ESNs_baseline[i][j]))) for j in 1:30] for i in 1:39]
er_srs_baseline = [[maximum(abs.(eigvals(er_ESNs_baseline[i][j]))) for j in 1:30] for i in 1:39]
cfg_srs_baseline = [[maximum(abs.(eigvals(cfg_ESNs_baseline[i][j]))) for j in 1:30] for i in 1:39]


using StatsPlots
b = boxplot([1],vcat(conn_srs_baseline[3:39]...),c=:dodgerblue4,label="Conn",  lw=3, alpha=0.8,boxwidth=0.2, whisker_width=0.15, )
boxplot!([2],vcat(er_srs_baseline[3:39]...),c=:crimson, label="ER", alpha=0.8,lw=3, legend=false,whisker_width=0.15,)
boxplot!([3], vcat(cfg_srs_baseline[3:39]...),c=:orange, label="CFG", lw=3, alpha=0.8,legend=false,whisker_width=0.15,)
plot!(xticks=([1,2,3],["Conn","ER","CFG"]), ylabel="Spectral Radius", legend=:topright, ylim=(0,4.5))
plot!(titlefontsize=18,legendfontsize=12,xtickfontsize=14,ylabelfontsize=16, grid=false, ytickfontsize=12,legend=false)

plot!(xticks=([1.5,4.5],["Larva","Adult"]), ylabel="Spectral Radius", title="Spectral Radius of Baseline Networks", legend=:topright, ylim=(0.5,1.8))
plot!(titlefontsize=18,legendfontsize=12,xtickfontsize=14,ylabelfontsize=16, ytickfontsize=12,legend=:bottom,grid=false)



connlarva_mean_srs = mean.(conn_srs_baseline[3:11])
connadult_mean_srs = mean.(conn_srs_baseline[12:39])
randlarva_mean_srs = mean.(er_srs_baseline[3:11])
randadult_mean_srs = mean.(er_srs_baseline[12:39])
cfglarva_mean_srs = mean.(cfg_srs_baseline[3:11])
cfgadult_mean_srs = mean.(cfg_srs_baseline[12:39])

pvalue(SignedRankTest(connlarva_mean_srs, randlarva_mean_srs))
pvalue(SignedRankTest(connadult_mean_srs, randadult_mean_srs))



using CSV, DataFrames

bar([1],[mean(conn_larva_baseline_sr)], c=dodgerblue_larva, label="Conn", alpha=0.8, yerror=std(conn_larva_baseline_sr)/sqrt(length(conn_larva_baseline_sr)), lw=2)
bar!([2],[mean(random_larva_baseline_sr)], c=crimson_larva, label="ER", alpha=0.8, yerror=std(random_larva_baseline_sr)/sqrt(length(random_larva_baseline_sr)), lw=2)
bar!([3],[mean(cfg_larva_baseline_sr)], c=orange_larva, label="CFG", alpha=0.8, yerror=std(cfg_larva_baseline_sr)/sqrt(length(cfg_larva_baseline_sr)), lw=2)





conn_weights = [filter(!iszero, vcat([vcat(conn_ESNs[j][i]...) for i in 1:30]...)) for j in 3:39]
er_weights = [filter(!iszero, vcat([vcat(er_ESNs[j][i]...) for i in 1:30]...)) for j in 3:39]
cfg_weights = [filter(!iszero, vcat([vcat(cfg_ESNs[j][i]...) for i in 1:30]...)) for j in 3:39]


conn_larva_sums = [[sum(abs.(conn_ESNs[j][i])) for i in 1:30] for j in 3:11]
er_larva_sums = [[sum(abs.(er_ESNs[j][i])) for i in 1:30] for j in 3:11]
cfg_larva_sums = [[sum(abs.(cfg_ESNs[j][i])) for i in 1:30] for j in 3:11]
conn_adult_sums = [[sum(abs.(conn_ESNs[j][i])) for i in 1:30] for j in 12:39]
er_adult_sums = [[sum(abs.(er_ESNs[j][i])) for i in 1:30] for j in 12:39]
cfg_adult_sums = [[sum(abs.(cfg_ESNs[j][i])) for i in 1:30] for j in 12:39]

mean(mean.(conn_larva_sums))
mean(mean.(er_larva_sums))
# mean(mean.(cfg_larva_sums))
mean(mean.(conn_adult_sums))
mean(mean.(er_adult_sums))
# mean(mean.(cfg_adult_sums))
pvalue(MannWhitneyUTest(vcat(conn_larva_sums...), vcat(er_larva_sums...)))
pvalue(MannWhitneyUTest(vcat(conn_adult_sums...), vcat(er_adult_sums...)))

h1 = histogram(vcat(conn_weights[3:11]...),bins=-3:0.2:3,normalize=true,c=:dodgerblue4, lw=0.5, alpha=0.8, label="Conn", xlabel="Weight", ylabel="Density", title="Distribution of Non-Zero Weights Across Larva Connectome Networks")
h2 = histogram(vcat(er_weights[3:11]...),bins=-3:0.2:3,normalize=true,c=:crimson, lw=0.2, alpha=0.8, label="ER", xlabel="Weight", ylabel="Density", title="Distribution of Non-Zero Weights Across Larva Connectome Networks")
histogram!(vcat(conn_weights[3:11]...),bins=-3:0.2:3,normalize=true,c=:dodgerblue4, lw=0.2, alpha=0.8, label="Conn", xlabel="Weight", ylabel="Density", title="Distribution of Non-Zero Weights Across Larva Connectome Networks")

plot(h1, h2, layout=(1,2), xlim=(-3,3), size=(700,400), margin=5mm,lw=0.1, grid=false, legendfontsize=10) #,xlim=(0.01,0.49))



b = boxplot(vcat(er_weights[1:9]...), c=:crimson, lw=3, alpha=0.8,legend=false,permute=(:x, :y), xaxis=false,yaxis=false,grid=false)
# boxplot!(vcat(cfg_weights[1:9]...),c=:orange, lw=3, alpha=0.8,legend=false,permute=(:x, :y))
boxplot!(vcat(conn_weights[1:9]...),c=:dodgerblue4, lw=3, alpha=0.8,permute=(:x, :y))
plot!(size=(600,150))


using Plots.PlotMeasures
s1 = stephist(vcat(conn_weights[1:9]...),bins=-3:0.1:3,normalize=true,c=:dodgerblue4, lw=5, label="Conn", xlabel="Weight", ylabel="Density", title="Distribution of Weights", legend=:topright)
stephist!(vcat(er_weights[1:9]...),bins=-3:0.2:3,normalize=true,c=:crimson, lw=5, label="ER", xlabel="Weight", ylabel="Density", title="Distribution of Non-Zero Weights Across Larva Connectome Networks")
# stephist!(vcat(cfg_weights[1:9]...),bins=-3:0.2:3,normalize=true,c=:orange, lw=5, label="CFG", xlabel="Weight", ylabel="Density", title="Distribution of Non-Zero Weights Across Larva Connectome Networks")
stephist!(vcat(conn_weights[1:9]...),bins=-3:0.1:3,normalize=true,c=:dodgerblue4, lw=4, label=false, xlabel="Weight", ylabel="Density", title="Distribution of Weights", legend=:topright)
plot!(grid=false, yticks=false,size=(500,500), margin=5mm, legendfontsize=12, xlim=(-3,3))
plot!(titlefontsize=18,legendfontsize=12,xtickfontsize=14,ylabelfontsize=16,xlabelfontsize=16)

s2 = stephist(vcat(conn_weights[12:end]...),bins=-3:0.1:3,normalize=true,c=dodgerblues[2], lw=5, label="Conn", xlabel="Weight", ylabel="Density", title="Distribution of Weights", legend=:topright)
stephist!(vcat(er_weights[12:end]...),bins=-3:0.2:3,normalize=true,c=crimsons[2], lw=5, label="ER", xlabel="Weight", ylabel="Density", title="Distribution of Non-Zero Weights Across Larva Connectome Networks")
# stephist!(vcat(cfg_weights[12:end]...),bins=-3:0.2:3,normalize=true,c=:orange, lw=5, label="CFG", xlabel="Weight", ylabel="Density", title="Distribution of Non-Zero Weights Across Larva Connectome Networks")
stephist!(vcat(conn_weights[12:end]...),bins=-3:0.1:3,normalize=true,c=dodgerblues[2], lw=4, label=false, xlabel="Weight", ylabel="Density", title="Distribution of Weights", legend=:topright)
plot!(grid=false, yticks=false,size=(500,500), margin=5mm, legendfontsize=12, xlim=(-3,3))
plot!(titlefontsize=18,legendfontsize=12,xtickfontsize=14,ylabelfontsize=16,xlabelfontsize=16)

s = plot(s1, s2, layout=(2,1), size=(600,800), margin=5mm)

maxval_conn = 0
for i in 3:39
    for j in 1:30
        maxval_conn = max(maxval_conn, maximum(abs.(conn_ESNs[i][j])))
    end
end
maxval_conn

maxval_er = 0
for i in 3:39
    for j in 1:30
        maxval_er = max(maxval_er, maximum(abs.(er_ESNs[i][j])))
    end
end
maxval_er
























######## node dists



    using StatsBase, Plots

function avg_degree_hist1(networks, range_i; nbins=50)
    all_degrees = Float64[]
    for i in range_i
        for j in 1:length(networks[i])
            A = networks[i][j]
            degs = sum(A .!= 0, dims=1)[:] .+ sum(A .!= 0, dims=2)[:]  # total degree
            append!(all_degrees, degs)
        end
    end

    hist = fit(Histogram, all_degrees, nbins=nbins, closed=:left)
    edges = hist.edges[1]
    counts = hist.weights
    normalized = counts ./ sum(counts)
    centres = (edges[1:end-1] .+ edges[2:end]) ./ 2
    widths = diff(edges)
    return centres, normalized, widths
end

# get histograms
conn_larva_centres, conn_larva_norm, conn_larva_widths = avg_degree_hist1(conn_ESNs, 3:11)
er_larva_centres, er_larva_norm, er_larva_widths = avg_degree_hist1(er_ESNs, 3:11)
cfg_larva_centres, cfg_larva_norm, cfg_larva_widths = avg_degree_hist1(cfg_ESNs, 3:11)

bar(conn_larva_centres, conn_larva_norm; width=conn_larva_widths, label="Conn", alpha=0.6, l1=.1,bar_width=1., color=:dodgerblue)
bar!(er_larva_centres, er_larva_norm; width=er_larva_widths, label="ER", bar_width=1., alpha=0.6, color=:crimson)


function avg_degree_hist2(networks, range_i; nbins=50, edges=nothing)
    all_degrees = Float64[]
    for i in range_i
        for j in 1:length(networks[i])
            A = networks[i][j]
            degs = sum(A .!= 0, dims=1)[:] .+ sum(A .!= 0, dims=2)[:]  # total degree
            append!(all_degrees, degs)
        end
    end

    if isnothing(edges)
        hist = fit(Histogram, all_degrees, nbins=nbins, closed=:left)
    else
        hist = fit(Histogram, all_degrees, edges, closed=:left)
    end

    edges = hist.edges[1]
    counts = hist.weights
    normalized = counts ./ sum(counts)
    centres = (edges[1:end-1] .+ edges[2:end]) ./ 2
    widths = diff(edges)
    return centres, normalized, widths, edges
end

# compute shared bin edges from one dataset (say, conn)
_, _, _, shared_edges = avg_degree_hist2(conn_ESNs, 12:39)

# now use those same edges for all
conn_centres, conn_norm, conn_widths, _ = avg_degree_hist2(conn_ESNs, 12:39, edges=shared_edges)
er_centres, er_norm, er_widths, _ = avg_degree_hist2(er_ESNs, 12:39, edges=shared_edges)
cfg_centres, cfg_norm, cfg_widths, _ = avg_degree_hist2(cfg_ESNs, 12:39, edges=shared_edges)

# plot
bar(conn_centres, conn_norm; label="Connectome", alpha=0.6, l1=.1, color=:dodgerblue)
bar!(er_centres, er_norm; label="ER", alpha=0.6, color=:crimson)
bar!(cfg_centres, cfg_norm; label="CFG", alpha=0.6, color=:orange)

xlabel!("Node degree")
ylabel!("Density")
title!("Adult network degree distributions", xlim=(0,30))


max_degs = []

for i in 3:39
    A = conn_ESNs[i][1]
    degs = sum(A .!= 0, dims=1)[:] .+ sum(A .!= 0, dims=2)[:]  # total degree
    max_deg = maximum(degs)

    push!(max_degs, max_deg)
end


sizes = [size(conn_ESNs[i][1], 1) for i in 3:39]
plot(max_degs ./ sizes, marker=:o, xlabel="Network ID")

