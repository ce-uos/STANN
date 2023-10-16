open_project stann_tb
add_files stann_testable.cpp
add_files -tb testbench/main_tb.cpp
open_solution "solution1" -flow_target vivado
csim_design
exit
