function lyapunov_spectrum_input(W, Win; 
                                 T=500, 
                                 Twarm=50, 
                                 input_range=0.5)

    N = size(W,1)

    x = randn(N); x ./= norm(x)
    Q = Matrix(I, N, N)
    λs = zeros(N)

    tmp = zeros(N)
    J = zeros(N, N)

    for t in 1:T

        u = (2rand() - 1) * input_range

        x = tanh.(W*x .+ Win*u)

        # J = Diagonal(1 .- x.^2) * W

        tmp .= 1 .- x.^2

        @inbounds for i in 1:N
            @simd for j in 1:N
                J[i,j] = tmp[i] * W[i,j]
            end
        end

        Q, R = qr(J * Q)

        if t > Twarm
            λs .+= log.(abs.(diag(R)) .+ eps())
        end

        Q = Matrix(Q)
    end

    return λs ./ (T - Twarm)
end


##########
# Testing how long we need T for convergence of the spectrum

        Ts = [100, 200, 300, 500, 1000]

        results = []
        Ww = deepcopy(conn_ESNs[3][1])
        Win = (rand(size(Ww,1)) .* 2 .- 1) .* 0.01
        for T in Ts
            λ = lyapunov_spectrum_input(Ww, Win; T=T, Twarm=round(Int,0.1T))
            push!(results, λ)
        end


        mles = [maximum(λ) for λ in results]
        dkys = [lyapunov_dimension(λ) for λ in results]

        plot(Ts, mles, marker=:o, label="MLE")
        plot(Ts, dkys, marker=:o, label="Dky")
###########



function fast_mle(W, Win; T=1000, Twarm=100)
    N = size(W,1)
    x = randn(N); x ./= norm(x)
    v = randn(N); v ./= norm(v)
    λsum = 0.0
    count = 0

    for t in 1:T
        u = (2rand() - 1) * 0.8
        x = tanh.(W*x .+ Win*u)

        v .= (1 .- x.^2) .* (W*v)

        if t > Twarm
            nv = norm(v)
            λsum += log(nv + eps())
            v ./= nv
            count += 1
        end
    end

    return λsum / count
end



function lyapunov_dimension(λs)
    λs = sort(λs; rev=true)
    s = cumsum(λs)
    k = findlast(>(0), s)
    if k === nothing
        return 0.0
    elseif k == length(λs)
        return float(k)
    else
        return k + s[k]/abs(λs[k+1])
    end
end


function local_lyapunov_spectrum(
    W, Win;
    T=1000,
    Twarm=100
)

    N = size(W,1)

    x = zeros(N)
    λsum = zeros(N)

    diagJ = similar(x)
    J = similar(W)

    for t in 1:T

        # drive
        u = (2rand() - 1) * 0.8
        x .= tanh.(W*x .+ Win*u)

        # compute diagonal term
        @inbounds @simd for i in 1:N
            diagJ[i] = 1 - x[i]^2
        end

        # scale rows of W into J
        @inbounds for i in 1:N
            @simd for j in 1:N
                J[i,j] = diagJ[i] * W[i,j]
            end
        end

        if t > Twarm
            eigvalsJ = eigvals!(J)
            λsum .+= log.(abs.(eigvalsJ) .+ eps())
        end
    end

    return λsum ./ (T - Twarm)
end

function analyze_ESNs_inputdriven(ESNs, rho_values, input_scalings)

    n_rhos = length(rho_values)
    n_input_scalings = length(input_scalings)

    λs_overall = Matrix{Any}(undef, n_rhos, n_input_scalings)
    Ds_overall = Matrix{Any}(undef, n_rhos, n_input_scalings)

    ρWs = [maximum(abs.(eigvals(W))) for W in ESNs]

    @threads for i_rho in 1:n_rhos
        rho = rho_values[i_rho]

        for i_input in 1:n_input_scalings
            input_scaling = input_scalings[i_input]

            λs_all = Float64[]
            Dky_all = Float64[]

            for (iW, W) in enumerate(ESNs)
            
                Wscaled = W * (rho / ρWs[iW])

                Win = (rand(size(Wscaled,1)) .* 2 .- 1) .* input_scaling

                spectrum = lyapunov_spectrum_input(Wscaled, Win; T=500, Twarm=50)
                push!(λs_all, maximum(spectrum))
                push!(Dky_all, lyapunov_dimension(spectrum))
            end

            λs_overall[i_rho, i_input] = λs_all
            Ds_overall[i_rho, i_input] = Dky_all
        end
    end

    mles_egs = [zeros(n_rhos, n_input_scalings) for _ in 1:length(ESNs)]
    for i in 1:length(ESNs)
        for j in 1:n_rhos
            for k in 1:n_input_scalings
                mles_egs[i][j,k] = λs_overall[j,k][i]
            end
        end
    end

    ds_egs = [zeros(n_rhos, n_input_scalings) for _ in 1:length(ESNs)]
    for i in 1:length(ESNs)
        for j in 1:n_rhos
            for k in 1:n_input_scalings
                ds_egs[i][j,k] = Ds_overall[j,k][i]
            end
        end
    end

    return mles_egs, ds_egs
end

using Main.Threads
nsrs = 30
spectral_radii = exp.(range(log(0.01), log(20.0), length=nsrs))
input_scalings = exp.(range(log(0.01), log(20.0), length=nsrs-1))
pushfirst!(input_scalings, 0.0)


conn_MLEs_total = []
conn_Ds_total = []
er_MLEs_total = []
er_Ds_total = []
cfg_MLEs_total = []
cfg_Ds_total = []

using ProgressLogging

@progress for id in 3:39
    println("Analyzing ESN $id...")
    λ_conn, D_conn = analyze_ESNs_inputdriven(conn_ESNs[id][1:2], spectral_radii, input_scalings)
    λ_er, D_er = analyze_ESNs_inputdriven(er_ESNs[id][1:2], spectral_radii, input_scalings)
    λ_cfg, D_cfg = analyze_ESNs_inputdriven(cfg_ESNs[id][1:2], spectral_radii, input_scalings)

    push!(conn_MLEs_total, λ_conn)
    push!(conn_Ds_total, D_conn)
    push!(er_MLEs_total, λ_er)
    push!(er_Ds_total, D_er)
    push!(cfg_MLEs_total, λ_cfg)
    push!(cfg_Ds_total, D_cfg)
end

max_mle_larva = 0
min_mle_larva = minimum([minimum(mean(mean.(conn_MLEs_total[1:9]))), minimum(mean(mean.(er_MLEs_total[1:9])))])
max_mle_adult = maximum([maximum(mean(mean.(conn_MLEs_total[10:end]))), maximum(mean(mean.(er_MLEs_total[10:end])))])
min_mle_adult = minimum([minimum(mean(mean.(conn_MLEs_total[10:end]))), minimum(mean(mean.(er_MLEs_total[10:end])))])

using NaNMath
using ColorSchemes

col = cgrad(:coolwarm,rev=true)
s_MLE_larva_conn = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(conn_MLEs_total[1:9])), 
        c=col, camera=(100, 20), zlim=(min_mle_larva,0.05), clim=(min_mle_larva,0.05),
        xticks=[-1,0,1], yticks=[-1, 0, 1])
s_MLE_larva_er = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(er_MLEs_total[1:9])), 
        c=col, camera=(100, 20), zlim=(min_mle_larva,0.05), clim=(min_mle_larva,0.05),
        xticks=[-1,0,1], yticks=[-1, 0, 1])
s_MLE_larva_cfg = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(cfg_MLEs_total[1:9])),
        c=col, camera=(100, 20), zlim=(min_mle_larva,0.05), clim=(min_mle_larva,0.05),
        xticks=[-1,0,1], yticks=[-1, 0, 1])
s_MLE_adult_conn = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(conn_MLEs_total[10:end])), 
        c=col, camera=(100, 20), zlim=(min_mle_adult,max_mle_adult), clim=(min_mle_adult,0.05),
        xticks=[-1,0,1], yticks=[-1, 0, 1])
s_MLE_adult_er = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(er_MLEs_total[10:end])), 
        c=col,camera=(100, 20), zlim=(min_mle_adult,max_mle_adult), clim=(min_mle_adult,0.05),
        xticks=[-1,0,1], yticks=[-1, 0, 1])
s_MLE_adult_cfg = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(cfg_MLEs_total[10:end])), 
        c=col,camera=(100, 20), zlim=(min_mle_adult,max_mle_adult), clim=(min_mle_adult,0.05),
        xticks=[-1,0,1], yticks=[-1, 0, 1])

p = plot(s_MLE_larva_conn, s_MLE_larva_er, s_MLE_adult_conn, s_MLE_adult_er, layout=(2,2), size=(900,800), title=["Conn ESN MLE Larva" "ER ESN MLE Larva" "Conn ESN MLE Adult" "ER ESN MLE Adult"])


p11 = plot(spectral_radii,mean(mean.(conn_MLEs_total[1:9]))[:,1], xscale=:log10, lw=6, c=:dodgerblue4, label="Conn ESN Larva")
plot!(spectral_radii,mean(mean.(er_MLEs_total[1:9]))[:,1], xlim=(0.05,20), xscale=:log10, lw=6, c=:crimson, label="ER ESN Larva")
plot!(spectral_radii,mean(mean.(cfg_MLEs_total[1:9]))[:,1], xlim=(0.05,20), xscale=:log10, lw=6, c=oranges[2], label="Conn ESN Adult")
hline!([0], label=false, linestyle=:dash, c=:black)

p12 = plot(spectral_radii,mean(mean.(conn_MLEs_total[10:end]))[:,1], xscale=:log10, lw=6, c=:dodgerblue4, label="Conn ESN Larva")
plot!(spectral_radii,mean(mean.(er_MLEs_total[10:end]))[:,1], xlim=(0.05,20),xscale=:log10, lw=6, c=:crimson, label="ER ESN Larva")
plot!(spectral_radii,mean(mean.(cfg_MLEs_total[10:end]))[:,1], xlim=(0.05,20), xscale=:log10, lw=6, c=oranges[3], label="ER ESN Adult")
hline!([0], label=false, linestyle=:dash, c=:black)

p21 = plot(spectral_radii,mean(mean.(conn_Ds_total[1:9]))[:,1], xscale=:log10, lw=6, c=:dodgerblue4, label="Conn ESN Larva")
plot!(spectral_radii,mean(mean.(er_Ds_total[1:9]))[:,1], xlim=(0.05,20),xscale=:log10, lw=6, c=:crimson, label="ER ESN Larva")
plot!(spectral_radii,mean(mean.(cfg_Ds_total[1:9]))[:,1], xlim=(0.05,20), xscale=:log10, lw=6, c=oranges[2], label="Conn ESN Adult")

p22 = plot(spectral_radii,mean(mean.(conn_Ds_total[10:end]))[:,1], xscale=:log10, lw=6, c=:dodgerblue4, label="Conn ESN Larva")
plot!(spectral_radii,mean(mean.(er_Ds_total[10:end]))[:,1], xlim=(0.05,20),xscale=:log10, lw=6, c=:crimson, label="ER ESN Larva")
plot!(spectral_radii,mean(mean.(cfg_Ds_total[10:end]))[:,1], xlim=(0.05,20), xscale=:log10, lw=6, c=oranges[3], label="Conn ESN Adult")


max_D_larva = maximum([maximum(mean(mean.(conn_Ds_total[1:9]))), maximum(mean(mean.(er_Ds_total[1:9])))])
min_D_larva = minimum([minimum(mean(mean.(conn_Ds_total[1:9]))), minimum(mean(mean.(er_Ds_total[1:9])))])
max_D_adult = maximum([maximum(mean(mean.(conn_Ds_total[10:end]))), maximum(mean(mean.(er_Ds_total[10:end])))])
min_D_adult = minimum([minimum(mean(mean.(conn_Ds_total[10:end]))), minimum(mean(mean.(er_Ds_total[10:end])))])

colo = :plasma
s_D_larva_conn = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(conn_Ds_total[1:9])), 
        c=colo, camera=(60, 20), xticks=[-1,0,1], yticks=[-1, 0, 1], zlim=(min_D_larva,max_D_larva), clim=(min_D_larva,max_D_larva))
s_D_larva_er = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(er_Ds_total[1:9])), 
        c=colo, xticks=[-1,0,1], yticks=[-1, 0, 1], camera=(60, 20), zlim=(min_D_larva,max_D_larva), clim=(min_D_larva,max_D_larva))
s_D_adult_conn = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(conn_Ds_total[10:end])), 
        c=colo,  camera=(60, 20), xticks=[-1,0,1], yticks=[-1, 0, 1], zlim=(min_D_adult,max_D_adult), clim=(min_D_adult,max_D_adult))    
s_D_adult_er = surface(log10.(input_scalings), log10.(spectral_radii), mean(mean.(er_Ds_total[10:end])), 
        c=colo, camera=(60, 20), xticks=[-1,0,1], yticks=[-1, 0, 1], zlim=(min_D_adult,max_D_adult), clim=(min_D_adult,max_D_adult))

p = plot(s_D_larva_conn, s_D_larva_er, s_D_adult_conn, s_D_adult_er, layout=(2,2), size=(900,800), title=["Conn ESN Dky Larva" "ER ESN Dky Larva" "Conn ESN Dky Adult" "ER ESN Dky Adult"], xlabel="Input Scaling", ylabel="Spectral Radius")


using Plots.PlotMeasures
MLE_plot_larva = plot(
heatmap(input_scalings1, spectral_radii, mean(mean.(conn_MLEs_total[1:9]))[:,2:end], xscale=:log10, yscale=:log10, c=:viridis),
heatmap(input_scalings1, spectral_radii, mean(mean.(er_MLEs_total[1:9]))[:,2:end], xscale=:log10, yscale=:log10, c=:viridis),
layout=(1,2), size=(900,400), clim=(-4,0), title=["Conn ESN Dky" "ER ESN Dky"], xlabel="Input Scaling", ylabel="Spectral Radius", margin=5mm
)

MLE_plot_adult = plot(
heatmap(input_scalings, spectral_radii, mean(mean.(conn_MLEs_total[10:end])), xscale=:log10, yscale=:log10, c=:viridis),
heatmap(input_scalings, spectral_radii, mean(mean.(er_MLEs_total[10:end])), xscale=:log10, yscale=:log10, c=:viridis),
layout=(1,2), size=(900,400), clim=(-4,0), title=["Conn ESN Dky" "ER ESN Dky"], xlabel="Input Scaling", ylabel="Spectral Radius", margin=5mm
)

D_plot_larva = plot(
heatmap(input_scalings, spectral_radii, mean(mean.(conn_Ds_total[1:9])), xscale=:log10, yscale=:log10, c=:viridis),
heatmap(input_scalings, spectral_radii, mean(mean.(er_Ds_total[1:9])), xscale=:log10, yscale=:log10, c=:viridis),
layout=(1,2), size=(900,400), clim=(0,1), title=["Conn ESN Dky" "ER ESN Dky"], xlabel="Input Scaling", ylabel="Spectral Radius", margin=5mm
)


D_plot_adult = plot(
heatmap(input_scalings, spectral_radii, mean(mean.(conn_Ds_total[10:end])), xscale=:log10, yscale=:log10, c=:viridis),
heatmap(input_scalings, spectral_radii, mean(mean.(er_Ds_total[10:end])), xscale=:log10, yscale=:log10, c=:viridis),
layout=(1,2), size=(900,400), clim=(0,16), title=["Conn ESN Dky" "ER ESN Dky"], xlabel="Input Scaling", ylabel="Spectral Radius", margin=5mm
)

plot(MLE_plot_larva, MLE_plot_adult, D_plot_larva, D_plot_adult, layout=(4,1), size=(1500,2000), title=["MLE Larva" "MLE Adult" "Dky Larva" "Dky Adult"])

function total_variation_input(LLE)
    tv = 0.0
    # Horizontal differences
    tv += NaNMath.sum(abs.(LLE[:, 2:end] .- LLE[:, 1:end-1]))
    # Vertical differences
    tv += NaNMath.sum(abs.(LLE[2:end, :] .- LLE[1:end-1, :]))
    return tv
end

tv_mle_conn = [[total_variation_input(conn_MLEs_total[i][j]) for j in 1:2] for i in 1:length(conn_MLEs_total)]
tv_mle_er = [[total_variation_input(er_MLEs_total[i][j]) for j in 1:2] for i in 1:length(er_MLEs_total)]
tv_dky_conn = [[total_variation_input(conn_Ds_total[i][j]) for j in 1:2] for i in 1:length(conn_Ds_total)]
tv_dky_er = [[total_variation_input(er_Ds_total[i][j]) for j in 1:2] for i in 1:length(er_Ds_total)]


using StatsPlots
b1 = bar([1], [mean(mean.(tv_mle_conn[1:9]))], yerror=std(mean.(tv_mle_conn[1:9]))/sqrt(9), label="Conn ESN Larva", color=:dodgerblue4)
bar!([1.5], [mean(mean.(tv_mle_conn[10:end]))], yerror=std(mean.(tv_mle_conn[10:end]))/sqrt(28), label="Conn ESN Adult", color=dodgerblues[3], legend=:topleft)
bar!([2.5], [mean(mean.(tv_mle_er[1:9]))], yerror=std(mean.(tv_mle_er[1:9]))/sqrt(9), label="ER ESN Larva", color=:crimson)
bar!([3], [mean(mean.(tv_mle_er[10:end]))], yerror=std(mean.(tv_mle_er[10:end]))/sqrt(28), ylim=(0,800), label="ER ESN Adult", color=crimsons[3])

b2 = bar([1], [mean(mean.(tv_dky_conn[1:9]))], yerror=std(mean.(tv_dky_conn[1:9]))/sqrt(9), label="Conn ESN Larva", color=:dodgerblue4)
bar!([1.5], [mean(mean.(tv_dky_conn[10:end]))], yerror=std(mean.(tv_dky_conn[10:end]))/sqrt(28), label="Conn ESN Adult", color=dodgerblues[3], legend=:topleft)
bar!([2.5], [mean(mean.(tv_dky_er[1:9]))], yerror=std(mean.(tv_dky_er[1:9]))/sqrt(9), label="ER ESN Larva", color=:crimson)
bar!([3], [mean(mean.(tv_dky_er[10:end]))], yerror=std(mean.(tv_dky_er[10:end]))/sqrt(28), ylim=(0,1400), label="ER ESN Adult", color=crimsons[3])

p = plot(b1, b2, layout=(1,2), size=(900,400), grid=false, title=["Total Variation MLE" "Total Variation Dky"], ylabel="Total Variation", xticks=([1, 1.5, 2.5, 3], ["Conn Larva", "Conn Adult", "ER Larva", "ER Adult"]))
