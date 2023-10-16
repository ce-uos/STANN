open_project stann_tb
add_files -tb testbench/training_tb.cpp
open_solution "solution1" -flow_target vivado
csim_design
exit
