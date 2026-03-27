
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

nsrs = 30
spectral_radii = exp.(range(log(0.01), log(20.0), length=nsrs))


sparsity_levels = [0.9, 0.95, 0.985]


networks = [[Matrix(Reservoirs_GraphTheory.create_reservoir(100, 1-S, 0.99)) for i in 1:100] for S in sparsity_levels]

mle_90, D_90 = analyze_ESNs_inputdriven(networks[1], spectral_radii, [0.0])
mle_95, D_95 = analyze_ESNs_inputdriven(networks[2], spectral_radii, [0.0])
mle_985, D_985 = analyze_ESNs_inputdriven(networks[3], spectral_radii, [0.0])

P1 = plot(spectral_radii, mean(mle_90[:,1]), label="S=0.9", 
    xscale=:log10, xlabel="Spectral radius", ylabel="MLE", title="MLE vs Spectral Radius for Different Sparsity Levels")
plot!(spectral_radii, mean(mle_95[:,1]), label="S=0.95")
plot!(spectral_radii, mean(mle_985[:,1]), label="S=0.985")
plot!(xlim=(0.05,20),)

P2 = plot(spectral_radii, mean(D_90[:,1]), label="S=0.9", 
    xscale=:log10, xlabel="Spectral radius", ylabel="Dky", title="Lyapunov Dimension vs Spectral Radius for Different Sparsity Levels")
plot!(spectral_radii, mean(D_95[:,1]), label="S=0.95")
plot!(spectral_radii, mean(D_985[:,1]), label="S=0.985")
plot!(xlim=(0.05,20),legend=false)

P3 = plot(P1, P2, tickfontsize=16, legendfontsize=16, layout=(1,2), lw=4, size=(900,400), grid=false, title=["Max Lyapunov Exponent" "Lyapunov Dimension"], margin=5mm)

