
using HypothesisTests, Statistics

# task 1
pvalue(SignedRankTest(mean.(conn_opt_performances[1][3:11]), mean.(er_opt_performances[1][3:11])))
pvalue(SignedRankTest(mean.(conn_opt_performances[1][12:end]), mean.(er_opt_performances[1][12:end])))
# task 2
pvalue(SignedRankTest(mean.(conn_opt_performances[2][3:11]), mean.(er_opt_performances[2][3:11])))
pvalue(SignedRankTest(mean.(conn_opt_performances[2][12:end]), mean.(er_opt_performances[2][12:end])))
# task 3
pvalue(SignedRankTest(mean.(conn_opt_performances[3][3:11]), mean.(er_opt_performances[3][3:11])))
pvalue(SignedRankTest(mean.(conn_opt_performances[3][12:end]), mean.(er_opt_performances[3][12:end])))
# task 4
pvalue(SignedRankTest(mean.(conn_opt_performances[4][3:11]), mean.(er_opt_performances[4][3:11])))
pvalue(SignedRankTest(mean.(conn_opt_performances[4][12:end]), mean.(er_opt_performances[4][12:end])))
# task 5
pvalue(SignedRankTest(mean.(conn_opt_performances[5][3:11]), mean.(er_opt_performances[5][3:11])))
pvalue(SignedRankTest(mean.(conn_opt_performances[5][12:end]), mean.(er_opt_performances[5][12:end])))
# task 6
pvalue(SignedRankTest(mean.(conn_opt_performances[6][3:11]), mean.(er_opt_performances[6][3:11])))
pvalue(SignedRankTest(mean.(conn_opt_performances[6][12:end]), mean.(er_opt_performances[6][12:end])))
# task 7
pvalue(SignedRankTest(mean.(conn_opt_performances[7][3:11]), mean.(er_opt_performances[7][3:11])))
pvalue(SignedRankTest(mean.(conn_opt_performances[7][12:end]), mean.(er_opt_performances[7][12:end])))
# task 8
pvalue(SignedRankTest(mean.(conn_opt_performances[8][3:11]), mean.(er_opt_performances[8][3:11])))
pvalue(SignedRankTest(mean.(conn_opt_performances[8][12:end]), mean.(er_opt_performances[8][12:end])))


########## normalise by wiring cost

# Task 1
pvalue(SignedRankTest(
    mean.(conn_opt_performances[1][3:11] ./ mean.(conn_costs[3:11])),
    mean.(er_opt_performances[1][3:11] ./ mean.(er_costs[3:11]))))
pvalue(SignedRankTest(
    mean.(conn_opt_performances[1][12:end] ./ mean.(conn_costs[12:end])),
    mean.(er_opt_performances[1][12:end] ./ mean.(er_costs[12:end]))))
# Task 2
pvalue(SignedRankTest(
    mean.(conn_opt_performances[2][3:11] ./ mean.(conn_costs[3:11])),
    mean.(er_opt_performances[2][3:11] ./ mean.(er_costs[3:11]))))
pvalue(SignedRankTest(
    mean.(conn_opt_performances[2][12:end] ./ mean.(conn_costs[12:end])),
    mean.(er_opt_performances[2][12:end] ./ mean.(er_costs[12:end]))))
# Task 3
pvalue(SignedRankTest(
    mean.(conn_opt_performances[3][3:11] ./ mean.(conn_costs[3:11])),
    mean.(er_opt_performances[3][3:11] ./ mean.(er_costs[3:11]))))
pvalue(SignedRankTest(
    mean.(conn_opt_performances[3][12:end] ./ mean.(conn_costs[12:end])),
    mean.(er_opt_performances[3][12:end] ./ mean.(er_costs[12:end]))))  
# Task 4
pvalue(SignedRankTest(
    mean.(conn_opt_performances[4][3:11] ./ mean.(conn_costs[3:11])),
    mean.(er_opt_performances[4][3:11] ./ mean.(er_costs[3:11]))))
pvalue(SignedRankTest(
    mean.(conn_opt_performances[4][12:end] ./ mean.(conn_costs[12:end])),
    mean.(er_opt_performances[4][12:end] ./ mean.(er_costs[12:end]))))
# Task 5
pvalue(SignedRankTest(
    mean.(conn_opt_performances[5][3:11] ./ mean.(conn_costs[3:11])),
    mean.(er_opt_performances[5][3:11] ./ mean.(er_costs[3:11]))))
pvalue(SignedRankTest(
    mean.(conn_opt_performances[5][12:end] ./ mean.(conn_costs[12:end])),
    mean.(er_opt_performances[5][12:end] ./ mean.(er_costs[12:end]))))
# Task 6
pvalue(SignedRankTest(
    mean.(conn_opt_performances[6][3:11] ./ mean.(conn_costs[3:11])),
    mean.(er_opt_performances[6][3:11] ./ mean.(er_costs[3:11]))))
pvalue(SignedRankTest(
    mean.(conn_opt_performances[6][12:end] ./ mean.(conn_costs[12:end])),
    mean.(er_opt_performances[6][12:end] ./ mean.(er_costs[12:end]))))
# Task 7
pvalue(SignedRankTest(
    mean.(conn_opt_performances[7][3:11] ./ mean.(conn_costs[3:11])),
    mean.(er_opt_performances[7][3:11] ./ mean.(er_costs[3:11]))))
pvalue(SignedRankTest(
    mean.(conn_opt_performances[7][12:end] ./ mean.(conn_costs[12:end])),
    mean.(er_opt_performances[7][12:end] ./ mean.(er_costs[12:end]))))  
# Task 8
pvalue(SignedRankTest(
    mean.(conn_opt_performances[8][3:11] ./ mean.(conn_costs[3:11])),
    mean.(er_opt_performances[8][3:11] ./ mean.(er_costs[3:11]))))
pvalue(SignedRankTest(
    mean.(conn_opt_performances[8][12:end] ./ mean.(conn_costs[12:end])),
    mean.(er_opt_performances[8][12:end] ./ mean.(er_costs[12:end]))))




################ neural correlations

SignedRankTest(mean.(conn_corrs)[3:11], mean.(rand_corrs)[3:11])
SignedRankTest(mean.(conn_corrs)[12:end], mean.(rand_corrs)[12:end])

