conn_mats = deepcopy(conn_ESNs)
er_mats = deepcopy(er_ESNs)
cfg_mats = deepcopy(cfg_ESNs)


conn_evals = [[eigvals(mat) for mat in conn_mats[i][1:30]] for i in 1:39]
er_evals = [[eigvals(mat) for mat in er_mats[i][1:30]] for i in 1:39]
cfg_evals = [[eigvals(mat) for mat in cfg_mats[i][1:30]] for i in 1:39]

using KernelDensity

function mean_kde_eigs(mats; bandwidth=0.1, gridsize=200)
    # Collect all eigenvalues for these N matrices
    eigs = vcat([eigvals(mat) for mat in mats]...)
    reals = real.(eigs)
    imags = imag.(eigs)

    # Compute KDEs
    kde_re = kde(reals; bandwidth=bandwidth)
    kde_im = kde(imags; bandwidth=bandwidth)

    return kde_re, kde_im
end




c_real_larva = [real.(vcat(conn_evals[3:11]...)[i]) for i in 1:length(vcat(conn_evals[3:11]...))]
c_real_adult = [real.(vcat(conn_evals[12:39]...)[i]) for i in 1:length(vcat(conn_evals[12:39]...))]
c_imag_larva = [imag.(vcat(conn_evals[3:11]...)[i]) for i in 1:length(vcat(conn_evals[3:11]...))]
c_imag_adult = [imag.(vcat(conn_evals[12:39]...)[i]) for i in 1:length(vcat(conn_evals[12:39]...))]
er_real_larva = [real.(vcat(er_evals[3:11]...)[i]) for i in 1:length(vcat(er_evals[3:11]...))]
er_real_adult = [real.(vcat(er_evals[12:39]...)[i]) for i in 1:length(vcat(er_evals[12:39]...))]
er_imag_larva = [imag.(vcat(er_evals[3:11]...)[i]) for i in 1:length(vcat(er_evals[3:11]...))]
er_imag_adult = [imag.(vcat(er_evals[12:39]...)[i]) for i in 1:length(vcat(er_evals[12:39]...))]
cfg_real_larva = [real.(vcat(cfg_evals[3:11]...)[i]) for i in 1:length(vcat(cfg_evals[3:11]...))]
cfg_real_adult = [real.(vcat(cfg_evals[12:39]...)[i]) for i in 1:length(vcat(cfg_evals[12:39]...))]
cfg_imag_larva = [imag.(vcat(cfg_evals[3:11]...)[i]) for i in 1:length(vcat(cfg_evals[3:11]...))]
cfg_imag_adult = [imag.(vcat(cfg_evals[12:39]...)[i]) for i in 1:length(vcat(cfg_evals[12:39]...))]

vec1 = vcat(cfg_real_adult...)
vec2 = vcat(cfg_imag_adult...)
df = DataFrame(RealPart = vec1, ImagPart = vec2)


using Plots.PlotMeasures
bins=-0.95:0.05:0.95
bins = unique(vcat(
    -0.95:0.05:-0.1,
    -0.1:0.01:0.1,
    0.1:0.05:0.95
))
s1 = stephist(vcat(c_real_larva...), bins=bins, normalize=true, lw=5, yscale=:log10, label="Conn", c=:dodgerblue4)
stephist!(vcat(er_real_larva...), bins=bins, grid=false, normalize=true, lw=5,  yscale=:log10, label="ER", c=:crimson)
s2 = stephist(vcat(c_imag_larva...), bins=bins, normalize=true, lw=5, label=false, yscale=:log10,legend=false, c=:dodgerblue4)
stephist!(vcat(er_imag_larva...), bins=bins, grid=false, normalize=true, lw=5,  yscale=:log10,label=false, c=:crimson)
s3 = stephist(vcat(c_real_adult...), bins=bins, normalize=true, lw=5, label="Conn", yscale=:log10, c=dodgerblues[3])
stephist!(vcat(er_real_adult...), bins=bins, grid=false, normalize=true,  yscale=:log10,lw=5, label="ER", c=crimsons[3])
s4 = stephist(vcat(c_imag_adult...), bins=bins, normalize=true, lw=5,  yscale=:log10,label=false,legend=false, c=dodgerblues[3])
stephist!(vcat(er_imag_adult...), bins=bins, grid=false, normalize=true, lw=5, yscale=:log10, label=false, c=crimsons[3])
pp = plot(s1,s2,s3,s4, layout=(2,2),size=(800,800), ylim=(1e-3,1e2), xtickfontsize=12, ytickfontsize=12, legendfontsize=12, xlabelfontsize=13, ylabelfontsize=13, margin=5mm)

# For each of the 39 sets
conn_kdes = [mean_kde_eigs(conn_mats[i]) for i in 1:39]
er_kdes   = [mean_kde_eigs(er_mats[i])   for i in 1:39]
cfg_kdes  = [mean_kde_eigs(cfg_mats[i])  for i in 1:39]

idd=3  # choose which of the 39 to look at
kde_re_conn, kde_im_conn = conn_kdes[idd]
kde_re_er,   kde_im_er   = er_kdes[idd]

p1 = plot(kde_re_conn.x, kde_re_conn.density, label="Conn", c=dodgerblues[1], linewidth=4)
plot!(kde_re_er.x, kde_re_er.density, label="ER", c=crimsons[2], linewidth=4)
xlabel!("Eigenvalue")
ylabel!("Density", title="Real eigenvalue distributions")

p2 = plot(kde_im_conn.x, kde_im_conn.density, label=false,legend=false, c=dodgerblues[1], linewidth=4)
plot!(kde_im_er.x, kde_im_er.density, label=false, c=crimsons[2], linewidth=4)
xlabel!("Eigenvalue")
ylabel!("Density", title="Imaginary eigenvalue distributions")

plot(p1, p2, layout=(2,1), size=(600,800), title="Eigenvalue Distributions for Subnetwork")

idd=12
s1 = scatter(real.(conn_evals[idd][1]), imag.(conn_evals[idd][1]),aspectratio=true, xlim=(-1.02,1.02), ylim=(-1,1),c=:dodgerblue4, xlabel="Real", ylabel="Imaginary", markerstrokewidth=0, label="Conn",legend=false, title="\n Connectome")
s2 = scatter(real.(er_evals[idd][1]), imag.(er_evals[idd][1]), aspectratio=true,xlim=(-1.02,1.02), ylim=(-1,1),c=:crimson, xlabel="Real", ylabel="Imaginary", markerstrokewidth=0, label="ER",legend=false, title="\n ER")
s22 = scatter(real.(cfg_evals[idd][1]), imag.(cfg_evals[idd][1]), aspectratio=true,xlim=(-1.02,1.02), ylim=(-1,1),c=oranges[2], xlabel="Real", ylabel="Imaginary", markerstrokewidth=0, label="CFG",legend=false, title="\n CFG")

pp = plot(s1,s2,layout=(1,2), grid=false, size=(800,400), margin=5mm, xlim=(-1.1,1.1), ylim=(-1.1,1.1), xtickfontsize=12, ytickfontsize=12, legendfontsize=12, xlabelfontsize=13, ylabelfontsize=13      )


ida = 15
s3 = scatter(real.(conn_evals[ida][1]), imag.(conn_evals[ida][1]),aspectratio=true, xlim=(-1.02,1.02), ylim=(-1,1),c=dodgerblues[3], xlabel="Real", ylabel="Imaginary", markerstrokewidth=0, label="Conn",legend=false, title="Example Eigenspectrum - Conn")
s4 = scatter(real.(er_evals[ida][1]), imag.(er_evals[ida][1]), aspectratio=true,xlim=(-1.02,1.02), ylim=(-1,1),c=crimsons[3], xlabel="Real", ylabel="Imaginary", markerstrokewidth=0, label="ER",legend=false, title="Example Eigenspectrum - ER")


using LinearAlgebra, KernelDensity, Interpolations, Statistics, Plots

# KDE for one group
function group_kde(mats; bandwidth=0.1, gridsize=200)
    eigs = vcat([eigvals(mat) for mat in mats]...)
    reals = real.(eigs)
    imags = imag.(eigs)
    kde_re = kde(reals; bandwidth=bandwidth)
    kde_im = kde(imags; bandwidth=bandwidth)
    return kde_re, kde_im
end

# KDEs for all groups
conn_kdes_larva = [group_kde(conn_mats[i][1:30]) for i in 3:11]
er_kdes_larva  = [group_kde(er_mats[i][1:30])   for i in 3:11]
cfg_kdes_larva = [group_kde(cfg_mats[i][1:30])   for i in 3:11]
conn_kdes_adult = [group_kde(conn_mats[i][1:30]) for i in 12:39]
er_kdes_adult  = [group_kde(er_mats[i][1:30])   for i in 12:39]
cfg_kdes_adult = [group_kde(cfg_mats[i][1:30])   for i in 12:39]

# Put them all on the same x-grid
xgrid = range(-1.2, 1.2; length=300)   # adjust range as needed
dens_re_conn_larva = zeros(length(xgrid), 9)
dens_im_conn_larva = zeros(length(xgrid), 9)
dens_re_er_larva = zeros(length(xgrid), 9)
dens_im_er_larva = zeros(length(xgrid), 9)
dens_re_cfg_larva = zeros(length(xgrid), 9)
dens_im_cfg_larva = zeros(length(xgrid), 9)
dens_re_conn_adult = zeros(length(xgrid), 28)
dens_im_conn_adult = zeros(length(xgrid), 28)
dens_re_er_adult = zeros(length(xgrid), 28)
dens_im_er_adult = zeros(length(xgrid), 28)
dens_re_cfg_adult = zeros(length(xgrid), 28)
dens_im_cfg_adult = zeros(length(xgrid), 28)


using Interpolations


for (j,(kde_re,kde_im)) in enumerate(conn_kdes_larva)
    f_re = LinearInterpolation(kde_re.x, kde_re.density,
                               extrapolation_bc=Interpolations.Line())
    f_im = LinearInterpolation(kde_im.x, kde_im.density,
                               extrapolation_bc=Interpolations.Line())
    dens_re_conn_larva[:,j] .= f_re.(xgrid)
    dens_im_conn_larva[:,j] .= f_im.(xgrid)
end
for (j,(kde_re,kde_im)) in enumerate(er_kdes_larva)
    f_re = LinearInterpolation(kde_re.x, kde_re.density,
                               extrapolation_bc=Interpolations.Line())
    f_im = LinearInterpolation(kde_im.x, kde_im.density,
                               extrapolation_bc=Interpolations.Line())
    dens_re_er_larva[:,j] .= f_re.(xgrid)
    dens_im_er_larva[:,j] .= f_im.(xgrid)
end
for (j,(kde_re,kde_im)) in enumerate(conn_kdes_adult)
    f_re = LinearInterpolation(kde_re.x, kde_re.density,
                               extrapolation_bc=Interpolations.Line())
    f_im = LinearInterpolation(kde_im.x, kde_im.density,
                               extrapolation_bc=Interpolations.Line())
    dens_re_conn_adult[:,j] .= f_re.(xgrid)
    dens_im_conn_adult[:,j] .= f_im.(xgrid)
end
for (j,(kde_re,kde_im)) in enumerate(er_kdes_adult)
    f_re = LinearInterpolation(kde_re.x, kde_re.density,
                               extrapolation_bc=Interpolations.Line())
    f_im = LinearInterpolation(kde_im.x, kde_im.density,
                               extrapolation_bc=Interpolations.Line())
    dens_re_er_adult[:,j] .= f_re.(xgrid)
    dens_im_er_adult[:,j] .= f_im.(xgrid)
end
for (j,(kde_re,kde_im)) in enumerate(cfg_kdes_larva)
    f_re = LinearInterpolation(kde_re.x, kde_re.density,
                               extrapolation_bc=Interpolations.Line())
    f_im = LinearInterpolation(kde_im.x, kde_im.density,
                               extrapolation_bc=Interpolations.Line())
    dens_re_cfg_larva[:,j] .= f_re.(xgrid)
    dens_im_cfg_larva[:,j] .= f_im.(xgrid)
end
for (j,(kde_re,kde_im)) in enumerate(cfg_kdes_adult)
    f_re = LinearInterpolation(kde_re.x, kde_re.density,
                               extrapolation_bc=Interpolations.Line())
    f_im = LinearInterpolation(kde_im.x, kde_im.density,
                               extrapolation_bc=Interpolations.Line())
    dens_re_cfg_adult[:,j] .= f_re.(xgrid)
    dens_im_cfg_adult[:,j] .= f_im.(xgrid)
end


# Average across the 39
mean_re_conn_larva = mean(dens_re_conn_larva, dims=2)
mean_im_conn_larva = mean(dens_im_conn_larva, dims=2)
mean_re_er_larva = mean(dens_re_er_larva, dims=2)
mean_im_er_larva = mean(dens_im_er_larva, dims=2)
mean_re_cfg_larva = mean(dens_re_cfg_larva, dims=2)
mean_im_cfg_larva = mean(dens_im_cfg_larva, dims=2)
mean_re_conn_adult = mean(dens_re_conn_adult, dims=2)
mean_im_conn_adult = mean(dens_im_conn_adult, dims=2)
mean_re_er_adult = mean(dens_re_er_adult, dims=2)
mean_im_er_adult = mean(dens_im_er_adult, dims=2)
mean_re_cfg_adult = mean(dens_re_cfg_adult, dims=2)
mean_im_cfg_adult = mean(dens_im_cfg_adult, dims=2)


mean_re_conn_larva

p1 = plot(xgrid, mean_re_conn_larva[:], label="Conn", c=:dodgerblue4, linewidth=4)
plot!(xgrid, mean_re_er_larva[:], label="ER", c=:crimson, linewidth=4)
# plot!(xgrid, mean_re_cfg_larva[:], label="CFG", c=oranges[2], linewidth=4)
xlabel!("Real Part")
ylabel!("Density")
p2 = plot(xgrid, mean_im_conn_larva[:], label=false,legend=false, c=:dodgerblue4, linewidth=4)
plot!(xgrid, mean_im_er_larva[:], label=false, c=:crimson, linewidth=4)
# plot!(xgrid, mean_im_cfg_larva[:], label=false, c=oranges[2], linewidth=4)
xlabel!("Imaginary Part")
ylabel!("Density")
p3 = plot(xgrid, mean_re_conn_adult[:], label="Conn", c=dodgerblues[3], linewidth=4)
plot!(xgrid, mean_re_er_adult[:], label="ER", c=crimsons[3], linewidth=4)
# plot!(xgrid, mean_re_cfg_adult[:], label="CFG", c=oranges[3], linewidth=4)
xlabel!("Real Part")
ylabel!("Density")
p4 = plot(xgrid, mean_im_conn_adult[:], label=false,legend=false, c=dodgerblues[3], linewidth=4)
plot!(xgrid, mean_im_er_adult[:], label=false, c=crimsons[3], linewidth=4)
# plot!(xgrid, mean_im_cfg_adult[:], label=false, c=oranges[3], linewidth=4)
xlabel!("Imaginary Part")
ylabel!("Density")

default(fontfamily="JuliaMono")

pp1 = plot(p1, p2, layout=(1,2), size=(600,300), grid=false, margin=3mm, plot_title="Mean Eigenvalue Distributions - Larva",titlefontsize=12)
pp2 = plot(p3, p4, layout=(1,2), size=(600,300), grid=false, margin=3mm, plot_title="Mean Eigenvalue Distributions - Adult")


pp3 = plot(pp1, pp2, layout=(2,1),xlim=(-1.1,1.1),size=(600,800), xtickfontsize=12, ytickfontsize=12, legendfontsize=12, xlabelfontsize=13, ylabelfontsize=13,yticks=[0,1,2,3], grid=false,dpi=600)



s = plot(s1,s2, grid=false, size=(600,400), xtickfontsize=12, xlim=(-1.1,1.1), ylim=(-1.1,1.1), ytickfontsize=12, legendfontsize=12, xlabelfontsize=13, ylabelfontsize=13, plot_title="Example Eigenspectra")













mean_conn_evals = [mean(abs.(conn_evals[i])) for i in 1:9]
mean_er_evals = [mean(abs.(er_evals[i])) for i in 1:9]
mean_cfg_evals = [mean(abs.(cfg_evals[i])) for i in 1:9]

se_conn_evals = [std(abs.(conn_evals[i]))/sqrt(length(conn_evals[i])) for i in 1:9]
se_er_evals = [std(abs.(er_evals[i]))/sqrt(length(er_evals[i])) for i in 1:9]
se_cfg_evals = [std(abs.(cfg_evals[i]))/sqrt(length(cfg_evals[i])) for i in 1:9]



using LaTeXStrings, Plots
mean_abs_evals = plot(mean_conn_evals, yerror=se_conn_evals, c=:blue,lw=3,m=:square,markerstrokecolor=:blue,label=false)
plot!(mean_er_evals, yerror=se_er_evals, c=:red,lw=3,m=:square,markerstrokecolor=:red, label=false)
plot!(mean_cfg_evals, yerror=se_cfg_evals, c=:orange,m=:square,markerstrokecolor=:orange,label=false, xticks=1:1:9, xlabel="Subnetwork ID", ylabel=L"\langle | \lambda_i | \rangle",lw=3, title="Mean of Absolute Eigenvalues")
scatter!([NaN],[NaN], m=:square, mc=:blue,label="Conn")
scatter!([NaN],[NaN], m=:square, mc=:red,label="ER")
scatter!([NaN],[NaN], m=:square, mc=:orange,label="CFG",grid=false)


conn_sc = [scatter(conn_evals[i], c=:blue, label="Conn", title="Subnetwork $i",xlabel="Re(x)", ylabel="Im(x)",leftmargin=5mm,bottommargin=5mm,legend=false) for i in 2:9]
er_sc = [scatter(er_evals[i], c=:red, label="ER", title="Subnetwork $i",xlabel="Re(x)", ylabel="Im(x)",leftmargin=5mm,bottommargin=5mm,legend=false) for i in 2:9]
cfg_sc = [scatter(cfg_evals[i], c=:orange, label="CFG",title="Subnetwork $i",xlabel="Re(x)", ylabel="Im(x)",leftmargin=5mm,bottommargin=5mm,legend=false) for i in 2:9]


pushfirst!(conn_sc, scatter(conn_evals[1], c=:blue, label="Conn", title="Subnetwork 1",xlabel="Re(x)", ylabel="Im(x)",leftmargin=5mm,bottommargin=5mm))
pushfirst!(er_sc, scatter(er_evals[1], c=:red, label="ER", title="Subnetwork 1",xlabel="Re(x)", ylabel="Im(x)",leftmargin=5mm,bottommargin=5mm))
pushfirst!(cfg_sc, scatter(cfg_evals[1], c=:orange, label="CFG", title="Subnetwork 1",xlabel="Re(x)", ylabel="Im(x)",leftmargin=5mm,bottommargin=5mm))



y1, y2 = rand(9), rand(9)
dy1, dy2 = 0.2rand(9), 0.2rand(9)
plot(y1, yerror=dy1, m=:square, lc=:reds, mc=:reds, msc=:red)
plot!(y2, yerror=dy2, m=:square, lc=:blues, mc=:blues, msc=:blues)
scatter!(y1, m=:square, mc=:reds, label=false)
scatter!(y2, m=:square, mc=:blues, label=false)









using Plots.PlotMeasures
plot(conn_sc[1], er_sc[1], cfg_sc[1], 
        conn_sc[2], er_sc[2], cfg_sc[2],
        conn_sc[3], er_sc[3], cfg_sc[3], 
        conn_sc[4], er_sc[4], cfg_sc[4], 
        conn_sc[5], er_sc[5], cfg_sc[5], 
        conn_sc[6], er_sc[6], cfg_sc[6], 
        conn_sc[7], er_sc[7], cfg_sc[7], 
        conn_sc[8], er_sc[8], cfg_sc[8], 
        conn_sc[9], er_sc[9], cfg_sc[9], 
        layout=(9,3),aspectratio=true,size=(700,3000),leftmargin=5mm, markerstrokewidth=0)





p = plot(conn_sc[1],conn_sc[2],conn_sc[3],conn_sc[4],conn_sc[5],conn_sc[6],conn_sc[7],conn_sc[8],conn_sc[9],
    er_sc[1],er_sc[2],er_sc[3],er_sc[4],er_sc[5],er_sc[6],er_sc[7],er_sc[8],er_sc[9],
    cfg_sc[1],cfg_sc[2],cfg_sc[3],cfg_sc[4],cfg_sc[5],cfg_sc[6],cfg_sc[7],cfg_sc[8],cfg_sc[9],
    layout=(3,9),size=(3000,1000),aspectratio=true,leftmargin=15mm,bottommargin=15mm,xlim=(-1.5,1.5),ylim=(-1.5,1.5),grid=false,markerstrokewidth=0)


p = plot(conn_sc[1],er_sc[1],cfg_sc[1],layout=(1,3),size=(1000,400),
leftmargin=5mm,bottommargin=5mm,aspectratio=true, markerstrokewidth=0,
grid=false,xticks=-1:0.5:1,xlim=(-1.1,1.1),legendmarkerwidth=0,dpi=500)


plot(conn_sc[1],er_sc[1],cfg_sc[1],reg_sc,layout=(2,2),size=(600,600))



e1 = plot(histogram(real.(conn_evals[1]),c=:blue,label="Conn",bins=-1:0.2:1,normalize=true,ylabel="Normalised frequency"),
        histogram(real.(er_evals[1]),c=:red, label="ER",bins=-1:0.2:1,normalize=true),
        histogram(real.(cfg_evals[1]),c=:orange, label="CFG",bins=-1:0.2:1,normalize=true),
        layout=(1,3), plot_title="Distribution of real parts of eigenvalues", size=(1000,1000),bottommargin=10mm, ylim=(0,3.5),fillalpha=0.7)

e2 = plot(histogram(imag.(conn_evals[1]),c=:blue,label="Conn",bins=-1:0.2:1,normalize=true,ylabel="Normalised frequency"),
        histogram(imag.(er_evals[1]),c=:red, label="ER",bins=-1:0.2:1,normalize=true), 
        histogram(imag.(cfg_evals[1]),c=:orange, label="CFG",bins=-1:0.2:1,normalize=true), 
        layout=(1,3), plot_title="Distribution of imaginary parts of eigenvalues", legend=false,ylim=(0,3.5),size=(1000,1000),fillalpha=0.7)


e1 = plot(histogram(real.(conn_evals[1]),c=:blue,label="Conn",bins=-1:0.2:1,normalize=true,ylabel="Normalised frequency"),
        histogram(real.(er_evals[1]),c=:red, label="ER",bins=-1:0.2:1,normalize=true),
        layout=(1,2), plot_title="Distribution of real parts of eigenvalues", size=(1000,1000),bottommargin=10mm, ylim=(0,3.5),fillalpha=0.7)


        e2 = plot(histogram(imag.(conn_evals[1]),c=:blue,label="Conn",bins=-1:0.2:1,normalize=true,ylabel="Normalised frequency"),
        histogram(imag.(er_evals[1]),c=:red, label="ER",bins=-1:0.2:1,normalize=true),
        layout=(1,2), plot_title="Distribution of imaginary parts of eigenvalues", legend=false,ylim=(0,3.5),size=(1000,1000),fillalpha=0.7)

e3 = plot(e1,e2, layout=(2,1),size=(1000,1000), yticks=0:4, 
legendfontsize=14,ylabelfontsize=14,tickfontsize=14, 
margin=5mm,grid=false,leftmargin=10mm,fillalpha=0.7, dpi=500)


s1 = scatter(real.(conn_evals[1]),c=:blue,label="Conn",markersize=5,markerstrokewidth=0)
s2 = scatter(real.(er_evals[1]),c=:red,label="ER",markersize=5,markerstrokewidth=0)
s3 = scatter(real.(cfg_evals[1]),c=:orange,label="CFG",markersize=5,markerstrokewidth=0)

s4 = scatter(sort(imag.(conn_evals[1])),c=:blue,label="Conn",markersize=5,markerstrokewidth=0)
s5 = scatter(sort(imag.(er_evals[1])),c=:red,label="ER",markersize=5,markerstrokewidth=0)
s6 = scatter(sort(imag.(cfg_evals[1])),c=:orange,label="CFG",markersize=5,markerstrokewidth=0)

p1 = plot(s1,s2,s3,layout=(1,3),plot_title="Real parts of eigenvalues")
p2 = plot(s4,s5,s6,layout=(1,3),plot_title="Imag parts of eigenvalues",legend=false)

p3 = plot(p1,p2,layout=(2,1), size=(1200,600), grid=false,markerstrokewidth=0)


s11 = scatter(sort(abs.(conn_evals[1])), c=:blue, label="Conn")
s12 = scatter(sort(abs.(er_evals[1])), c=:red, label="ER")

s21 = scatter(sort(abs.(conn_evals[2])), c=:blue, label="Conn")
s22 = scatter(sort(abs.(er_evals[2])), c=:red, label="ER")

s31 = scatter(sort(abs.(conn_evals[3])), c=:blue, label="Conn")
s32 = scatter(sort(abs.(er_evals[3])), c=:red, label="ER")

s41 = scatter(sort(abs.(conn_evals[4])), c=:blue, label="Conn")
s42 = scatter(sort(abs.(er_evals[4])), c=:red, label="ER")

s51 = scatter(sort(abs.(conn_evals[5])), c=:blue, label="Conn")
s52 = scatter(sort(abs.(er_evals[5])), c=:red, label="ER")

s61 = scatter(sort(abs.(conn_evals[6])), c=:blue, label="Conn")
s62 = scatter(sort(abs.(er_evals[6])), c=:red, label="ER")

s71 = scatter(sort(abs.(conn_evals[7])), c=:blue, label="Conn")
s72 = scatter(sort(abs.(er_evals[7])), c=:red, label="ER")

s81 = scatter(sort(abs.(conn_evals[8])), c=:blue, label="Conn")
s82 = scatter(sort(abs.(er_evals[8])), c=:red, label="ER")

s91 = scatter(sort(abs.(conn_evals[9])), c=:blue, label="Conn")
s92 = scatter(sort(abs.(er_evals[9])), c=:red, label="ER")




p11 = plot(s11,s21,s31,s41,s51,s61,s71,s81,s91,
                s12,s22,s32,s42,s52,s62,s72,s82,s92,
layout=(2,9), plot_title="Scatter plot of abs vals of eigenvalues",size=(1500,800),legend=false,markersize=5,markerstrokewidth=0)


