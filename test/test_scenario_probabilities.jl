module TestScenarioProbabilities

using Test
using LightBenders
using JuMP
using HiGHS

# Newsvendor with non-uniform scenario probabilities.
#
# The optimal order quantity of a newsvendor is a quantile of the demand
# distribution, so shifting probability mass between scenarios moves the
# optimum. That makes it a sharp test: if the probabilities were ignored (the
# behaviour before this feature), every case below would return the same answer.
#
# buy 5, sell 10, return 1. Ordering one more unit gains 10 if it sells and 1 if
# it does not, against a cost of 5, so it is worth ordering up to the point where
# P(demand >= q) * 10 + P(demand < q) * 1 = 5, i.e. the 4/9 upper quantile of
# demand.

Base.@kwdef mutable struct Inputs
    buy_price::Real
    sell_price::Real
    return_price::Real
    max_storage::Int
    demand::Vector{<:Real}
end

function state_variables_builder(inputs, stage)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    sp = LightBenders.SubproblemModel(model)
    if stage == 1
        @variable(sp, 0 <= bought <= inputs.max_storage)
        LightBenders.set_first_stage_state(sp, :bought, bought)
    elseif stage == 2
        @variable(sp, bought)
        LightBenders.set_second_stage_state(sp, :bought, bought)
    end
    return sp
end

function first_stage_builder(sp, inputs)
    bought = sp[:bought]
    @constraint(sp, bought <= inputs.max_storage)
    @objective(sp, Min, bought * inputs.buy_price)
    return sp
end

function second_stage_builder(sp, inputs)
    bought = sp[:bought]
    @variable(sp, dem in MOI.Parameter(0.0))
    @variable(sp, sold >= 0)
    @variable(sp, returned >= 0)
    @constraint(sp, sold <= dem)
    @constraint(sp, sold + returned <= bought)
    @objective(sp, Min, -sold * inputs.sell_price - returned * inputs.return_price)
    return sp
end

function second_stage_modifier(sp, inputs, s)
    JuMP.set_parameter_value(sp[:dem], inputs.demand[s])
    return nothing
end

function solve_with(probabilities; cut_strategy = LightBenders.CutStrategy.MultiCut)
    inputs = Inputs(5, 10, 1, 100, [10, 20, 30])
    options = LightBenders.PolicyTrainingOptions(;
        num_scenarios = length(inputs.demand),
        scenario_probabilities = probabilities,
        lower_bound = -1e6,
        implementation_strategy = LightBenders.SerialTraining(),
        stopping_rule = [
            LightBenders.GapWithMinimumNumberOfIterations(; abstol = 1e-2, min_iterations = 2),
            # Always cap: a gap rule alone loops forever if the bounds never meet,
            # which is exactly the failure a change to scenario weighting could cause.
            LightBenders.IterationLimit(50),
        ],
        cut_strategy = cut_strategy,
        verbose = false,
    )
    policy = LightBenders.train(;
        state_variables_builder, first_stage_builder,
        second_stage_builder, second_stage_modifier,
        inputs, policy_training_options = options,
    )
    de = LightBenders.deterministic_equivalent(;
        state_variables_builder, first_stage_builder,
        second_stage_builder, second_stage_modifier,
        inputs,
        options = LightBenders.DeterministicEquivalentOptions(;
            num_scenarios = length(inputs.demand),
            scenario_probabilities = probabilities,
        ),
    )
    return policy, de["objective", 0]
end

"An empty probability vector must reproduce the old uniform behaviour exactly."
function test_empty_means_uniform()
    p_empty, obj_empty = solve_with(Float64[])
    p_unif, obj_unif = solve_with([1 / 3, 1 / 3, 1 / 3])
    @test LightBenders.lower_bound(p_empty) ≈ LightBenders.lower_bound(p_unif) atol = 1e-6
    @test LightBenders.upper_bound(p_empty) ≈ LightBenders.upper_bound(p_unif) atol = 1e-6
    @test obj_empty ≈ obj_unif atol = 1e-6
    # Known optimum of the equiprobable newsvendor
    @test LightBenders.upper_bound(p_empty) ≈ -70 atol = 1.0
    return nothing
end

"""
Skewing probability towards the low-demand scenario must lower the optimal order
quantity, and towards the high-demand scenario must raise it. If probabilities
were ignored these would all coincide.
"""
function test_probabilities_change_the_solution()
    _, obj_low = solve_with([0.8, 0.1, 0.1])    # demand 10 is likely
    _, obj_mid = solve_with([1 / 3, 1 / 3, 1 / 3])
    _, obj_high = solve_with([0.1, 0.1, 0.8])   # demand 30 is likely

    # Objectives are profits (negated), and a distribution concentrated on high
    # demand is worth strictly more than one concentrated on low demand.
    @test obj_high < obj_mid < obj_low
    return nothing
end

"Benders must agree with the deterministic equivalent under non-uniform weights."
function test_benders_matches_de_nonuniform()
    for probs in ([0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.5, 0.2, 0.3])
        policy, de_obj = solve_with(probs)
        @test LightBenders.upper_bound(policy) ≈ de_obj atol = 1e-1
        @test LightBenders.lower_bound(policy) <= de_obj + 1e-6
    end
    return nothing
end

"Single-cut aggregation is probability-weighted too, so it must agree with multi-cut."
function test_single_cut_matches_multi_cut()
    probs = [0.6, 0.25, 0.15]
    p_multi, _ = solve_with(probs; cut_strategy = LightBenders.CutStrategy.MultiCut)
    p_single, de_obj = solve_with(probs; cut_strategy = LightBenders.CutStrategy.SingleCut)
    @test LightBenders.upper_bound(p_multi) ≈ de_obj atol = 1e-1
    @test LightBenders.upper_bound(p_single) ≈ de_obj atol = 1e-1
    return nothing
end

"Malformed probability vectors are rejected rather than silently misused."
function test_validation()
    inputs = Inputs(5, 10, 1, 100, [10, 20, 30])
    bad = (
        [0.5, 0.5],            # wrong length
        [0.5, 0.6, 0.1],       # does not sum to 1
        [-0.1, 0.6, 0.5],      # negative
    )
    for probs in bad
        options = LightBenders.PolicyTrainingOptions(;
            num_scenarios = 3, scenario_probabilities = probs, verbose = false,
        )
        @test_throws ErrorException LightBenders.validate_benders_training_options(options)
    end
    # CVaR with non-uniform probabilities is refused: the reformulation assumes
    # equiprobable scenarios.
    cvar_options = LightBenders.PolicyTrainingOptions(;
        num_scenarios = 3,
        scenario_probabilities = [0.6, 0.2, 0.2],
        risk_measure = LightBenders.CVaR(; alpha = 0.5, lambda = 0.5),
        verbose = false,
    )
    @test_throws ErrorException LightBenders.validate_benders_training_options(cvar_options)
    # ... but CVaR with uniform probabilities is fine.
    ok_options = LightBenders.PolicyTrainingOptions(;
        num_scenarios = 3,
        scenario_probabilities = [1 / 3, 1 / 3, 1 / 3],
        risk_measure = LightBenders.CVaR(; alpha = 0.5, lambda = 0.5),
        verbose = false,
    )
    @test LightBenders.validate_benders_training_options(ok_options) === nothing
    return nothing
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return nothing
end

TestScenarioProbabilities.runtests()

end # module
