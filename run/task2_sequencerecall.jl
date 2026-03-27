using .Reservoirs_src

# Define the task parameters
L = 20  # Length of the memorized sequence to recall
input_dim = 2
output_dim = 1
res_size = 300  # Number of neurons in the reservoir
n_trials_train = 400  # Number of training trials
n_trials_test = 100  # Number of testing trials
regularization_coefficient = 1e-6
input_scaling = 0.02
sparsity = 0.1
spectral_radius = 0.99
leak_rate = 1.0

function create_data(L)
    x1 = vcat([rand() for i in 1:L],[0 for i in 1:L+1])
    x2 = vcat([0 for i in 1:L],[1],[0 for i in 1:L])
    x3 = vcat([0 for i in 1:L+1],x1[1:L])
    return x1,x2,x3
end

L_min = 30   # Minimum sequence length
L_max = 30  # Maximum sequence length

function new_create_data(L_min, L_max)
    # Randomly select sequence length within the range [L_min, L_max]
    L = rand(L_min:L_max)
    
    # Create input x1 with a random sequence followed by zeros
    x1 = vcat([rand() for i in 1:L], [0 for i in 1:L+1])
    
    # Create cue input x2 with a single 1 at a random position
    x2 = vcat([0 for i in 1:L], [1], [0 for i in 1:L])
    
    # Create output x3 which is the delayed copy of the random sequence from x1
    x3 = vcat([0 for i in 1:L+1], x1[1:L])
    
    return x1, x2, x3
end


reservoir = Reservoirs_src.create_reservoir(res_size, sparsity, spectral_radius)

input_weights = Reservoirs_src.create_input_weights(input_dim, res_size, input_scaling)
output_weights = Reservoirs_src.initialize_output_weights(res_size, output_dim)

  

x1_train_data = []
x2_train_data = []
x3_train_targets = []

for x in 1:n_trials_train
    x1,x2,x3 = create_data(L)
    push!(x1_train_data, x1)
    push!(x2_train_data, x2)
    push!(x3_train_targets,x3)
end

x1_train_data = vcat(x1_train_data...)
x2_train_data = vcat(x2_train_data...)
x3_train_targets = vcat(x3_train_targets...)


input_data = hcat(x1_train_data,x2_train_data)
input_data = input_data'


reservoir_state_trains = zeros(res_size,length(x1_train_data))
current_state = zeros(res_size) 

for t in 1:length(x1_train_data)
    current_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, current_state, input_data[:,t], leak_rate)
    reservoir_state_trains[:, t] = current_state
end

output_weights = Reservoirs_src.train_output_weights(reservoir_state_trains, x3_train_targets', regularization_coefficient)
training_output = output_weights' * reservoir_state_trains


true_out = plot(x3_train_targets[1:200],lw=3,legend=false,ylim=(0,1),xlim=(0,100),title="True")

res_out = plot(training_output[1:200],lw=3,xlim=(0,100),xlabel="Time",legend=false,color=:red,title="Reservoir output")

pplot = plot(true_out, res_out,layout=(2,1))

x1_test_data = []
x2_test_data = []
x3_test_targets = []

for x in 1:n_trials_test
    x1,x2,x3 = create_data(L)
    push!(x1_test_data, x1)
    push!(x2_test_data, x2)
    push!(x3_test_targets,x3)
end

x1_test_data = vcat(x1_test_data...)
x2_test_data = vcat(x2_test_data...)
x3_test_targets = vcat(x3_test_targets...)


input_data_test = hcat(x1_test_data,x2_test_data)
input_data_test = input_data_test'


last_state = reservoir_state_trains[:, end]
final_outputs = zeros(output_dim, length(x3_test_targets))
reservoir_internal_states_recall = zeros(res_size, length(x3_test_targets))

for t in 1:length(x1_test_data)
    next_state = Reservoirs_src.update_reservoir_state(reservoir, input_weights, last_state, input_data_test[:,t], leak_rate)

    predicted_output = output_weights' * next_state
    reservoir_internal_states_recall[:, t] = next_state

    # Storing the predicted output
    final_outputs[t] = predicted_output
    
    # Updating the last state for the next prediction
    last_state = next_state
end

plot(final_outputs[1:100])
plot!(x3_test_targets[1:100])

final_outputs
n_trials_test
recall_score = cor(x3_test_targets,final_outputs').^2
recall_score = recall_score[1,1]

function recall_accuracy(ŷ, y, ε=0.05)
    correct = abs.(ŷ .- y) .< ε
    return sum(correct) / length(y)
end

function recall_accuracy_ignore_fixation(ŷ, y, L, ε=0.1)
    ŷ_mod = copy(ŷ)
    n_blocks = length(ŷ) ÷ L
    for i in 1:2:n_blocks  # blocks (fixation periods)
        start_idx = (i - 1)*L + 1
        end_idx = i*L
        ŷ_mod[start_idx:end_idx] .= 0
    end
    
    # compute accuracy ignoring those zeroed blocks
    mask = ŷ_mod .!= 0
    correct = abs.(ŷ_mod[mask] .- y[mask]) .< ε
    return sum(correct) / length(correct)
end


recall_score = recall_accuracy_ignore_fixation(vec(final_outputs), vec(x3_test_targets), L)

true_out = plot(x3_test_targets[21:200],lw=3,ylim=(-0.1,1.3),legend=false,xlim=(0,100),title="True")

res_out = plot(final_outputs[1:200],lw=3,ylim=(-0.1,1.3),xlim=(0,100),xlabel="Time",legend=false,color=:red,title="Reservoir output")
annotate!(10,0.9, text("Fixation", color=:green, font="Computer Modern"))
annotate!(30,1.25, text("Response", color=:orange, font="Computer Modern"))
plot!([0,20],[.7,0.7], lw=3, c=:green)
plot!([21,41],[1.1,1.1], lw=3, c=:orange)


using Plots.PlotMeasures
pplot = plot(true_out, res_out,layout=(2,1), fontfamily="JuliaMono", tick_direction=:out, xlabelfontsize=16,  size=(1000,450), ylim=(-.05,1.3), margin=5mm, bottommargin=5mm,grid=false, yticks=[0,1],titlefontsize=18, ytickfontsize=12, xtickfontsize=14)

