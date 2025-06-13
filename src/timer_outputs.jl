const to_train = TimerOutput()
const to_simulate = TimerOutput()

function save_train_timer(path::String)
    # Save the timer output to a file
    open(path, "w") do io
        return print_timer(io, to_train; sortby = :firstexec)
    end
end

function save_simulate_timer(path::String)
    # Save the timer output to a file
    open(path, "w") do io
        return print_timer(io, to_simulate; sortby = :firstexec)
    end
end