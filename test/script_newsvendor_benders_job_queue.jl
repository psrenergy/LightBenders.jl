module TestNewsvendorBendersJobQueue

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

function state_variables_builder(inputs)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    sp = LightBenders.SubproblemModel(model)
    # state variable
    @variable(sp, 0 <= bought <= inputs.max_storage)
    LightBenders.set_state(sp, :bought, bought)
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
    @objective(sp, Min, - sold * inputs.sell_price - returned * inputs.return_price)
    return sp
end

function second_stage_modifier(sp, inputs, s)
    dem = sp[:dem]
    JuMP.set_parameter_value(dem, inputs.demand[s])
    return nothing
end

function newsvendor_benders(;cut_strategy = LightBenders.CutStrategy.MultiCut)
    inputs = Inputs(5, 10, 1, 100, [10, 20, 30])
    num_scenarios = length(inputs.demand)

    policy_training_options = LightBenders.PolicyTrainingOptions(;
        num_scenarios=num_scenarios,
        lower_bound = -1e6,
        implementation_strategy = LightBenders.JobQueueTraining(),
        stopping_rule = LightBenders.GapWithMinimumNumberOfIterations(;abstol = 1e-1, min_iterations = 2),
        cut_strategy = cut_strategy
    )

    policy = LightBenders.train(;
        state_variables_builder,
        first_stage_builder,
        second_stage_builder,
        second_stage_modifier,
        inputs = inputs,
        policy_training_options
    )

    if LightBenders.JQM.is_worker_process()
        return nothing
    end

    @test LightBenders.lower_bound(policy) ≈ -70
    @test LightBenders.upper_bound(policy) ≈ -70

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
        )
    )

    @test results["objective", 0] ≈ -70 atol = 1e-2
end

function test_newsvendor_benders()
    @testset "[Job Queue] Benders Newsvendor single cut" begin
        newsvendor_benders(;cut_strategy = LightBenders.CutStrategy.SingleCut)
    end
    @testset "[Job Queue] Benders Newsvendor multi cut" begin
        newsvendor_benders(;cut_strategy = LightBenders.CutStrategy.MultiCut)
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

TestNewsvendorBendersJobQueue.runtests()

end # module TestNewsvendorBendersJobQueue