open_project stann_tb
add_files -tb testbench/singlelayer_tb.cpp
open_solution "solution1" -flow_target vivado
csim_design
exit
