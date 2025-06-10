module TestNewsvendorBenders

using Test
using LightBenders
using JuMP
using HiGHS

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
    # state variable
    if stage == 1
        @variable(sp, 0 <= bought <= inputs.max_storage)
        LightBenders.set_first_stage_state(sp, :bought, bought)
    elseif stage == 2
        @variable(sp, bought)
        LightBenders.set_second_stage_state(sp, :bought, bought)
    end
    return sp
end

function state_variables_builder_parameter(inputs, stage)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    sp = LightBenders.SubproblemModel(model)
    # state variable
    if stage == 1
        @variable(sp, 0 <= bought <= inputs.max_storage)
        LightBenders.set_first_stage_state(sp, :bought, bought)
    elseif stage == 2
        @variable(sp, bought in MOI.Parameter(0.0))
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
    @constraint(sp, sold_dem_con, sold <= dem)
    @constraint(sp, balance, sold + returned <= bought)
    @objective(sp, Min, -sold * inputs.sell_price - returned * inputs.return_price)
    return sp
end

function second_stage_modifier(sp, inputs, s)
    dem = sp[:dem]
    JuMP.set_parameter_value(dem, inputs.demand[s])
    return nothing
end

function newsvendor_benders(;
    cut_strategy = LightBenders.CutStrategy.MultiCut,
    risk_measure = LightBenders.RiskNeutral(),
    state_variables_builder = state_variables_builder,
    cut_relaxation = LightBenders.CutRelaxationOptions(active = false),
)
    inputs = Inputs(5, 10, 1, 100, [10, 20, 30])
    num_scenarios = length(inputs.demand)

    policy_training_options = LightBenders.PolicyTrainingOptions(;
        num_scenarios = num_scenarios,
        lower_bound = -1e6,
        implementation_strategy = LightBenders.SerialTraining(),
        stopping_rule = [LightBenders.GapWithMinimumNumberOfIterations(;
            abstol = 1e-1,
            min_iterations = 2,
        )],
        cut_strategy = cut_strategy,
        cut_relaxation = cut_relaxation,
        risk_measure = risk_measure,
        # debugging_options = LightBenders.DebuggingOptions(;
        #     logs_dir= joinpath(@__DIR__, "logs"),
        #     write_lp = true,
        # ),
    )

    policy = LightBenders.train(;
        state_variables_builder,
        first_stage_builder,
        second_stage_builder,
        second_stage_modifier,
        inputs = inputs,
        policy_training_options,
    )

    results = LightBenders.simulate(;
        state_variables_builder,
        first_stage_builder,
        second_stage_builder,
        second_stage_modifier,
        inputs,
        policy,
        simulation_options = LightBenders.SimulationOptions(
            policy_training_options;
            implementation_strategy = LightBenders.BendersSerialSimulation(),
            # debugging_options = LightBenders.DebuggingOptions(;
            #     logs_dir= joinpath(@__DIR__, "logs"),
            #     write_lp = true,
            # ),
        ),
    )

    return policy, results
end

function newsvendor_deterministic()
    inputs = Inputs(5, 10, 1, 100, [10, 20, 30])
    num_scenarios = length(inputs.demand)

    options = LightBenders.DeterministicEquivalentOptions(; num_scenarios = num_scenarios)

    det_eq_results = LightBenders.deterministic_equivalent(;
        state_variables_builder,
        first_stage_builder,
        second_stage_builder,
        second_stage_modifier,
        inputs,
        options,
    )

    @test det_eq_results["objective", 0] ≈ -70 atol = 1e-2
end

function test_newsvendor_benders()
    @testset "Benders Newsvendor single cut risk neutral" begin
        policy, results = newsvendor_benders(; cut_strategy = LightBenders.CutStrategy.SingleCut)
        @test LightBenders.lower_bound(policy) ≈ -70
        @test LightBenders.upper_bound(policy) ≈ -70
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
    @testset "Benders Newsvendor single cut risk neutral state parameters" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.SingleCut,
            state_variables_builder = state_variables_builder_parameter,
        )
        @test LightBenders.lower_bound(policy) ≈ -70
        @test LightBenders.upper_bound(policy) ≈ -70
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
    @testset "Benders Newsvendor single cut risk neutral with cut relaxation" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.SingleCut,
            cut_relaxation = LightBenders.CutRelaxationOptions(active = true),
        )
        @test LightBenders.lower_bound(policy) ≈ -70
        @test LightBenders.upper_bound(policy) ≈ -70
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
    @testset "Benders Newsvendor multi cut risk neutral" begin
        policy, results = newsvendor_benders(; cut_strategy = LightBenders.CutStrategy.MultiCut)
        @test LightBenders.lower_bound(policy) ≈ -70
        @test LightBenders.upper_bound(policy) ≈ -70
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
    @testset "Benders Newsvendor multi cut risk neutral state parameters" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.MultiCut,
            state_variables_builder = state_variables_builder_parameter,
        )
        @test LightBenders.lower_bound(policy) ≈ -70
        @test LightBenders.upper_bound(policy) ≈ -70
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
    @testset "Benders Newsvendor multi cut risk neutral with cut relaxation" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.MultiCut,
            cut_relaxation = LightBenders.CutRelaxationOptions(active = true),
        )
        @test LightBenders.lower_bound(policy) ≈ -70
        @test LightBenders.upper_bound(policy) ≈ -70
        @test results["objective", 0] ≈ -70 atol = 1e-2
    end
    @testset "Benders Newsvendor single cut CVaR" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.SingleCut,
            risk_measure = LightBenders.CVaR(alpha = 0.9, lambda = 0.5),
        )
        @test LightBenders.lower_bound(policy) ≈ -50
        @test LightBenders.upper_bound(policy) ≈ -50
        @test results["objective", 0] ≈ -50 atol = 1e-2
    end
    @testset "Benders Newsvendor single cut CVaR with cut relaxation" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.SingleCut,
            risk_measure = LightBenders.CVaR(alpha = 0.9, lambda = 0.5),
            cut_relaxation = LightBenders.CutRelaxationOptions(active = true),
        )
        @test LightBenders.lower_bound(policy) ≈ -50
        @test LightBenders.upper_bound(policy) ≈ -50
        @test results["objective", 0] ≈ -50 atol = 1e-2
    end
    @testset "Benders Newsvendor single cut CVaR state parameters" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.SingleCut,
            risk_measure = LightBenders.CVaR(alpha = 0.9, lambda = 0.5),
            state_variables_builder = state_variables_builder_parameter,
        )
        @test LightBenders.lower_bound(policy) ≈ -50
        @test LightBenders.upper_bound(policy) ≈ -50
        @test results["objective", 0] ≈ -50 atol = 1e-2
    end
    @testset "Benders Newsvendor multi cut CVaR" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.MultiCut,
            risk_measure = LightBenders.CVaR(alpha = 0.9, lambda = 0.5),
        )
        @test LightBenders.lower_bound(policy) ≈ -50
        @test LightBenders.upper_bound(policy) ≈ -50
        @test results["objective", 0] ≈ -50 atol = 1e-2
    end
    @testset "Benders Newsvendor multi cut CVaR with cut relaxation" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.MultiCut,
            risk_measure = LightBenders.CVaR(alpha = 0.9, lambda = 0.5),
            cut_relaxation = LightBenders.CutRelaxationOptions(active = true),
        )
        @test LightBenders.lower_bound(policy) ≈ -50
        @test LightBenders.upper_bound(policy) ≈ -50
        @test results["objective", 0] ≈ -50 atol = 1e-2
    end
    @testset "Benders Newsvendor multi cut CVaR state parameters" begin
        policy, results = newsvendor_benders(;
            cut_strategy = LightBenders.CutStrategy.MultiCut,
            risk_measure = LightBenders.CVaR(alpha = 0.9, lambda = 0.5),
            state_variables_builder = state_variables_builder_parameter,
        )
        @test LightBenders.lower_bound(policy) ≈ -50
        @test LightBenders.upper_bound(policy) ≈ -50
        @test results["objective", 0] ≈ -50 atol = 1e-2
    end
    @testset "Deterministic equivalent Newsvendor" begin
        newsvendor_deterministic()
    end
end

function runtests()
    Base.GC.gc()
    Base.GC.gc()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
end

TestNewsvendorBenders.runtests()

end # module TestNewsvendorBenders

# Equivalent code in SDDP.jl
# using SDDP, JuMP, HiGHS
# function test_bender_sddp()
#     buy_price = 5
#     sell_price = 10
#     return_price = 1
#     max_storage = 100
#     demand = 10:10:30
#     model = SDDP.LinearPolicyGraph(;
#         stages = 2,
#         sense = :Min,
#         lower_bound = -1e3,
#         optimizer = HiGHS.Optimizer,
#     ) do subproblem, stage
#         @variable(subproblem, 0 <= bought, SDDP.State, initial_value = 0.0)
#         if stage == 1
#             @constraint(subproblem, bought.out <= max_storage) 
#             @stageobjective(
#                 subproblem,
#                 bought.out * buy_price,
#             )
#         else
#             @variable(subproblem, sold >= 0)
#             @variable(subproblem, returned >= 0)
#             SDDP.parameterize(subproblem, demand) do s
#                 JuMP.set_upper_bound(sold, s)
#             end
#             @constraint(subproblem, balance, sold + returned <= bought.in)
#             @stageobjective(
#                 subproblem,
#                 - sold * sell_price - returned * return_price,
#             )
#         end
#     end
#     det = SDDP.deterministic_equivalent(model, HiGHS.Optimizer)
#     set_silent(det)
#     JuMP.optimize!(det)
#     @show JuMP.termination_status(det)
#     @show JuMP.objective_value(det)
# end
