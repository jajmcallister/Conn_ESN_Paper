using .Reservoirs_src, .ConnectomeFunctions, .ReservoirTasks
using LinearAlgebra, SparseArrays, OrdinaryDiffEq
using Statistics, Random
using DataFrames, CSV
using Dates
using FileIO, JLD2
using Base.Threads

#load these in loadfiles.jl
conn_esns
rand_esns
cfg_esns


function find_opt_params(avgres,task)
    best_performance = 0.0
    best_indx = 0
    for i in 1:length(avgres)
        if avgres[i].perf[task] > best_performance
            best_performance = avgres[i].perf[task]
            best_indx = i
        end
    end
    prams = []
    push!(prams, avgres[best_indx].sr)
    push!(prams, avgres[best_indx].iscale)
    push!(prams, avgres[best_indx].reg)
    push!(prams, avgres[best_indx].leak)

    return prams
end

function sweep_parameters_parallel(
    reservoirs_base, 
    hyperparams,
    num_sim_steps::Int,
    biases, variances,
    osc_data, lotka_data,
    lorenz_train_data, lorenz_test_data,
    rossler_train_data, rossler_test_data
)

    results = Vector{Any}(undef, length(hyperparams))
    avg_results = Vector{Any}(undef, length(hyperparams))

    Threads.@threads for i in 1:length(hyperparams)
        hp = hyperparams[i]
        sr = hp.sr
        iscale = hp.is
        reg = hp.rc
        leak = hp.alpha

        reservoirs = scale_matrices(reservoirs_base, sr)

        p_mem, e_mem = ReservoirTasks.res_performance_memory(reservoirs, num_sim_steps, iscale, reg, leak)
        p_recall, e_recall = ReservoirTasks.res_performance_recall(reservoirs, num_sim_steps, iscale, reg, leak)
        p_dec, e_dec = ReservoirTasks.res_performance_decisionmaking(reservoirs, num_sim_steps, biases, iscale, reg, leak)
        p_dd, e_dd = ReservoirTasks.res_performance_delay_decisionmaking(reservoirs, num_sim_steps, variances, iscale, reg, leak)
        p_osc, e_osc = ReservoirTasks.res_performance_oscillator(reservoirs, osc_data, 0.5, num_sim_steps, iscale, reg, leak)
        p_lot, e_lot = ReservoirTasks.res_performance_lotka(reservoirs, lotka_data, 0.5, num_sim_steps, iscale, reg, leak)
        p_lor, e_lor = ReservoirTasks.res_performance_lorenz(reservoirs, lorenz_train_data, lorenz_test_data, num_sim_steps, 0.5, iscale, reg, leak)
        p_ros, e_ros = ReservoirTasks.res_performance_rossler(reservoirs, rossler_train_data, rossler_test_data, num_sim_steps, 0.5, iscale, reg, leak)

        results[i] = (
            sr = sr, iscale = iscale, reg = reg, leak = leak,
            perf = (mem = p_mem, recall = p_recall, dec = p_dec, dd = p_dd,
                    osc = p_osc, lot = p_lot, lor = p_lor, ros = p_ros),
            err = (mem = e_mem, recall = e_recall, dec = e_dec, dd = e_dd,
                   osc = e_osc, lot = e_lot, lor = e_lor, ros = e_ros)
        )

        avg_results[i] = (
            sr = sr, iscale = iscale, reg = reg, leak = leak,
            perf = (mem = mean(p_mem), recall = mean(p_recall), dec = mean(p_dec), dd = mean(p_dd),
                    osc = mean(p_osc), lot = mean(p_lot), lor = mean(p_lor), ros = mean(p_ros)),
            err = (mem = mean(e_mem), recall = mean(e_recall), dec = mean(e_dec), dd = mean(e_dd),
                   osc = mean(e_osc), lot = mean(e_lot), lor = mean(e_lor), ros = mean(e_ros))
        )
    end

    return results, avg_results
end



num_res = 30


conn_results, conn_avg_results = sweep_parameters_parallel(
    deepcopy(conn_esns[net_id][1:num_res]),
    hyperparams,
    num_sim_steps,
    biases, variances,
    osc_data, lotka_data,
    lorenz_train_data, lorenz_test_data,
    rossler_train_data, rossler_test_data
)

er_results, er_avg_results = sweep_parameters_parallel(
    deepcopy(rand_esns[net_id][1:num_res]),
    hyperparams,
    num_sim_steps,
    biases, variances,
    osc_data, lotka_data,
    lorenz_train_data, lorenz_test_data,
    rossler_train_data, rossler_test_data
)

cfg_results, cfg_avg_results = sweep_parameters_parallel(
    deepcopy(cfg_esns[net_id][1:num_res]),
    hyperparams,
    num_sim_steps,
    biases, variances,
    osc_data, lotka_data,
    lorenz_train_data, lorenz_test_data,
    rossler_train_data, rossler_test_data
)

opt_mem_params_conn = find_opt_params(conn_avg_results,1)
opt_rec_params_conn = find_opt_params(conn_avg_results,2)
opt_dm_params_conn = find_opt_params(conn_avg_results,3)
opt_ddm_params_conn = find_opt_params(conn_avg_results,4)
opt_osc_params_conn = find_opt_params(conn_avg_results,5)
opt_lot_params_conn = find_opt_params(conn_avg_results,6)
opt_lor_params_conn = find_opt_params(conn_avg_results,7)
opt_ros_params_conn = find_opt_params(conn_avg_results,8)

conn_opts = [opt_mem_params_conn, opt_rec_params_conn, opt_dm_params_conn, opt_ddm_params_conn,
                                opt_osc_params_conn, opt_lot_params_conn, opt_lor_params_conn, opt_ros_params_conn]


opt_mem_params_er = find_opt_params(er_avg_results,1)
opt_rec_params_er = find_opt_params(er_avg_results,2)
opt_dm_params_er = find_opt_params(er_avg_results,3)
opt_ddm_params_er = find_opt_params(er_avg_results,4)
opt_osc_params_er = find_opt_params(er_avg_results,5)
opt_lot_params_er = find_opt_params(er_avg_results,6)
opt_lor_params_er = find_opt_params(er_avg_results,7)
opt_ros_params_er = find_opt_params(er_avg_results,8)

opt_mem_params_cfg = find_opt_params(cfg_avg_results,1)
opt_rec_params_cfg = find_opt_params(cfg_avg_results,2)
opt_dm_params_cfg = find_opt_params(cfg_avg_results,3)
opt_ddm_params_cfg = find_opt_params(cfg_avg_results,4)
opt_osc_params_cfg = find_opt_params(cfg_avg_results,5)
opt_lot_params_cfg = find_opt_params(cfg_avg_results,6)
opt_lor_params_cfg = find_opt_params(cfg_avg_results,7)
opt_ros_params_cfg = find_opt_params(cfg_avg_results,8)


opt_params_conn = [opt_mem_params_conn, opt_rec_params_conn, opt_dm_params_conn, opt_ddm_params_conn,
                                opt_osc_params_conn, opt_lot_params_conn, opt_lor_params_conn, opt_ros_params_conn]
opt_params_er = [opt_mem_params_er, opt_rec_params_er, opt_dm_params_er, opt_ddm_params_er,
                                opt_osc_params_er, opt_lot_params_er, opt_lor_params_er, opt_ros_params_er]
opt_params_cfg = [opt_mem_params_cfg, opt_rec_params_cfg, opt_dm_params_cfg, opt_ddm_params_cfg,
                                opt_osc_params_cfg, opt_lot_params_cfg, opt_lor_params_cfg, opt_ros_params_cfg]

