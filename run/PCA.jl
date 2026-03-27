using LinearAlgebra
using .Reservoirs_src
using ColorSchemes


##################
# PCA of activity states

nummm = 30

conn_evals = Vector{Any}(undef, 39)
rand_evals = Vector{Any}(undef, 39)
cfg_evals = Vector{Any}(undef, 39)
conn_corrs = Vector{Any}(undef, 39)
rand_corrs = Vector{Any}(undef, 39)
cfg_corrs = Vector{Any}(undef, 39)

using Main.Threads
@threads for id in 1:39
    evals1 = []
    evals2 = []
    evals3 = []
    corrs1 = []
    corrs2 = []
    corrs3 = []

    for idd in 1:nummm
        train_length = 1000
        leak_rate = 1.0

        reservoir1 = deepcopy(conn_ESNs[id][idd])
        reservoir2 = deepcopy(er_ESNs[id][idd])
        reservoir3 = deepcopy(cfg_ESNs[id][idd])
        res_size = size(reservoir1, 1)
        in_scaling = 0.01
        input_size = 1
        input_weights = Reservoirs_src.create_input_weights(input_size, res_size, in_scaling)
        # uniform noise input
        X_train = rand(train_length) .- 0.5

        # Run each reservoir
        function run_reservoir(reservoir)
            state = zeros(res_size, train_length)
            current = zeros(res_size)
            for t in 1:train_length
                current = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current, X_train[t], leak_rate)
                state[:, t] = current
            end
            return state
        end

        state1 = run_reservoir(reservoir1)
        state2 = run_reservoir(reservoir2)
        state3 = run_reservoir(reservoir3)

        _, evals_1, _ = Reservoirs_src.pca(state1, num_components=15)
        _, evals_2, _ = Reservoirs_src.pca(state2, num_components=15)
        _, evals_3, _ = Reservoirs_src.pca(state3, num_components=15)

        corr1 = (sum(abs.(cor(state1'))) - res_size)/(res_size^2 - res_size)
        corr2 = (sum(abs.(cor(state2'))) - res_size)/(res_size^2 - res_size)
        corr3 = (sum(abs.(cor(state3'))) - res_size)/(res_size^2 - res_size)

        push!(evals1, evals_1)
        push!(evals2, evals_2)
        push!(evals3, evals_3)
        push!(corrs1, corr1)
        push!(corrs2, corr2)
        push!(corrs3, corr3)
    end

    conn_evals[id] = evals1
    rand_evals[id] = evals2
    cfg_evals[id] = evals3
    conn_corrs[id] = corrs1
    rand_corrs[id] = corrs2
    cfg_corrs[id] = corrs3
end

using StatsPlots

# neural correlations
b = boxplot([1], mean.(conn_corrs[3:11]), c=:dodgerblue4)
boxplot!([2], mean.(rand_corrs[3:11]), c=:crimson)
boxplot!([3], mean.(cfg_corrs[3:11]), c=oranges[2])
boxplot!([5], mean.(conn_corrs[12:39]), c=dodgerblues[3])
boxplot!([6], mean.(rand_corrs[12:39]), c=crimsons[3])
boxplot!([7], mean.(cfg_corrs[12:39]), c=oranges[3], ylim=(0.2,0.7), grid=false, legend=false)

#############################
# normalise by sum
p1 = plot(mean(mean(conn_evals[1:2])) / sum(mean(mean(conn_evals[1:2]))), lw=3, c=:blue, markerstrokewidth=0, label="Connectome", fillalpha=0.1, markershape=:circle, markersize=5)
plot!(mean(mean(rand_evals[1:2])) / sum(mean(mean(rand_evals[1:2]))), lw=3, c=:red,markerstrokewidth=0, label="ER", ribbon=std(mean(rand_evals[1:2]))./sqrt(nummm), fillalpha=0.1, markershape=:circle, markersize=5)
# plot!(mean(mean(cfg_evals[1:2])) / sum(mean(mean(cfg_evals[1:2]))), lw=3, c=:darkorange, markerstrokewidth=0, label="CFG", ylabel="Normalised Eigenvalue", xlabel="Principal Component", title="C Elegans", ribbon=std(mean(cfg_evals[1:2]))./sqrt(nummm), fillalpha=0.1 , markershape=:circle, markersize=5)
plot!(mean(mean(conn_evals[1:2])) / sum(mean(mean(conn_evals[1:2]))), lw=3, c=:blue, markerstrokewidth=0, label=false, ribbon=std(mean(conn_evals[1:2]))./sqrt(nummm), fillalpha=0.1, markershape=:circle, markersize=5)


p2 = plot(mean(mean(conn_evals[3:11])) / sum(mean(mean(conn_evals[3:11]))), lw=3, c=:dodgerblue4, markerstrokewidth=0, ribbon=std(mean(conn_evals[3:11]))./sqrt(nummm),  fillalpha=0.1, label="Conn", markershape=:circle, markersize=5)
plot!(mean(mean(rand_evals[3:11])) / sum(mean(mean(rand_evals[3:11]))), lw=3, c=:crimson, markerstrokewidth=0, ribbon=std(mean(rand_evals[3:11]))./sqrt(nummm), fillalpha=0.1, label="ER", markershape=:circle, markersize=5)
# plot!(mean(mean(cfg_evals[3:11])) / sum(mean(mean(cfg_evals[3:11]))), lw=3, c=oranges[2],markerstrokewidth=0, xlabel="Principal Component", title="Larval Drosophila", ribbon=std(mean(cfg_evals[3:11]))./sqrt(nummm), fillalpha=0.1, label=false, markershape=:circle, markersize=5)
plot!(mean(mean(conn_evals[3:11])) / sum(mean(mean(conn_evals[3:11]))), lw=3, leftmargin=10mm , ylabelfontsize=14, ylabel="Proportion of \n Variance Explained \n", c=:dodgerblue4, markerstrokewidth=0, xlabel="Principal Component", title="Larval Drosophila", ribbon=std(mean(conn_evals[3:11]))./sqrt(nummm), fillalpha=0.1, label=false, markershape=:circle, markersize=5)

p3 = plot(mean(mean(conn_evals[12:39])) / sum(mean(mean(conn_evals[12:39]))), lw=3, c=dodgerblues[3], markerstrokewidth=0.1, fillalpha=0.9, label="Adult", markershape=:circle, markersize=5)
plot!(mean(mean(rand_evals[12:39])) / sum(mean(mean(rand_evals[12:39]))), lw=3, c=crimsons[3], markerstrokewidth=0.1, ribbon=std(mean(rand_evals[12:39]))./sqrt(nummm), fillalpha=0.1, label="ER", legend=false, markershape=:circle, markersize=5)
# plot!(mean(mean(cfg_evals[12:39])) / sum(mean(mean(cfg_evals[12:39]))), lw=3, c=oranges[3],markerstrokewidth=0, xlabel="Principal Component", title="Adult Drosophila", ribbon=std(mean(cfg_evals[12:39]))./sqrt(nummm), fillalpha=0.1, label=false, markershape=:circle, markersize=5)
plot!(mean(mean(conn_evals[12:39])) / sum(mean(mean(conn_evals[12:39]))), lw=3, c=dodgerblues[3], markerstrokewidth=0.1,xlabel="Principal Component",  title="Adult Drosophila", ribbon=std(mean(conn_evals[12:39]))./sqrt(nummm), fillalpha=0.1, label=false, markershape=:circle, markersize=5)

pp3 = plot(p2,p3,layout=(1,2),size=(700,400), legend=:bottomright, plot_title="Dimensionality of Dynamics", titlefontsize=16, grid=false, legendfontsize=14,bottommargin=5mm, xtickfontsize=14, ytickfontsize=14, xlabelfontsize=16, ylabelfontsize=16)


# Cumulative
# Larva
conn_p  = mean(mean(conn_evals[3:11]))
conn_p ./= sum(conn_p)
conn_cum = cumsum(conn_p)

rand_p  = mean(mean(rand_evals[3:11]))
rand_p ./= sum(rand_p)
rand_cum = cumsum(rand_p)

cfg_p = mean(mean(cfg_evals[3:11]))
cfg_p ./= sum(cfg_p)
cfg_cum = cumsum(cfg_p)

p2 = plot(conn_cum, lw=3, c=:dodgerblue4,
    ribbon = std(mean(conn_evals[3:11])) ./ sqrt(nummm),
    markershape=:circle, markersize=5,
    label="Conn")

plot!(rand_cum, lw=3, c=:crimson,
    ribbon = std(mean(rand_evals[3:11])) ./ sqrt(nummm),
    markershape=:circle, markersize=5,
    label="ER")
plot!(cfg_cum, lw=3, c=:darkorange,
    ribbon = std(mean(cfg_evals[3:11])) ./ sqrt(nummm),
    markershape=:circle, markersize=5,
    label="CFG")
plot!(conn_cum, lw=3, c=:dodgerblue4, xlabel="PC",
    ylabel="Cumulative Variance Explained", title="Larva",
    label=false)

    # Adult
conn_pA = mean(mean(conn_evals[12:39]))
conn_pA ./= sum(conn_pA)
conn_cumA = cumsum(conn_pA)

rand_pA = mean(mean(rand_evals[12:39]))
rand_pA ./= sum(rand_pA)
rand_cumA = cumsum(rand_pA)

cfg_pA = mean(mean(cfg_evals[12:39]))
cfg_pA ./= sum(cfg_pA)
cfg_cumA = cumsum(cfg_pA)

p3 = plot(conn_cumA, lw=3, c=dodgerblues[3], title="Adult",
    markershape=:circle, markersize=5,
    label="Conn")

plot!(rand_cumA, lw=3, c=crimsons[3],
    markershape=:circle, markersize=5,
    label="ER")

plot!(cfg_cumA, lw=3, c=oranges[3],
    markershape=:circle, markersize=5,
    label="CFG")
plot!(xlabel="PC",
    ylabel="Cumulative Variance Explained", title="Adult",
    label=false)


pp3 = plot(p2,p3,layout=(1,2), size=(700,400), legend=:bottomright, plot_title="Dimensionality of Dynamics", titlefontsize=16, grid=false, legendfontsize=14,leftmargin=5mm,  bottommargin=3mm, xtickfontsize=14, ytickfontsize=14, xlabelfontsize=16, ylabelfontsize=16)


function PR_per_network(evals; K=15)
    [participation_ratio11(
        mean(evals[i])[1:K] ./ sum(mean(evals[i])[1:K])
     ) for i in eachindex(evals)]
end

PR_conn_larva = PR_per_network(conn_evals[3:11])
PR_rand_larva = PR_per_network(rand_evals[3:11])

PR_conn_adult = PR_per_network(conn_evals[12:39])
PR_rand_adult = PR_per_network(rand_evals[12:39])

mean(PR_conn_larva), mean(PR_rand_larva)
mean(PR_conn_adult), mean(PR_rand_adult)
