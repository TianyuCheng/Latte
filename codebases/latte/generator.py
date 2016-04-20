#!/usr/bin/env python2
'''
    Latte Compiler 
'''

import os, sys
import compiler.ast
import inspect, compiler
from ast_matcher import *
from templates import *
import py_compile

NARGS = 3

def usage():
    usage_str = ""
    usage_str += "Usage: \n"
    usage_str += "\tpython [program_file(.py)] [out_cfile] \n"
    print usage_str
    return

LATTE_H_PATH = '''"Latte.h"'''

def make_include_header():
    header = ""
    header += '''#include ''' + LATTE_H_PATH 
    return header

def make_main_header():
    main_header = "int main (int argn, char** argv) { "
    return main_header

def make_newlines(num=1):
    return "\n" * num

# def make_indent(num=indent):
#     return "    " * num

def make_mkl_malloc(mat_name, dim_x, dim_y):
    return "double* %s = init_mkl_mat (%s, %s);" % (mat_name, dim_x, dim_y)

def make_mkl_free(mat_name): return "mkl_free(%s);" % (mat_name)

def make_FC_weights_init(name, dim_x, dim_y, prev_dim_x, prev_dim_y):
    declare_str = "vector<vector<double*>> %s (%s, %s);\n" % (name, dim_x, dim_y)
    init_str = "init_weights_mats(%s, %s, %s);" % (name, prev_dim_x, prev_dim_y)
    return declare_str + init_str

def make_FC_weights_free(name):
    return "free_weights_mats(%s);" % (name)

# input list of ensembles name
def make_allocate_block(ensembles_info):
    allocate_block = ["// Allocating memory for Output, Grad_output Matrices"]
    for enm in ensembles_info:
        output_mat_name = enm[0]+"_output"
        output_dim_x = enm[0]+".dim_x"
        output_dim_y = enm[0]+".dim_y"
        output_malloc_str = make_mkl_malloc(output_mat_name, output_dim_x, output_dim_y)
        allocate_block.append(output_malloc_str) 
    for enm in ensembles_info[:-1]:
        grad_mat_name = enm[0]+"_grad_output"
        grad_dim_x = enm[0]+".next_dim_x"
        grad_dim_y = enm[0]+".next_dim_y"
        grad_malloc_str = make_mkl_malloc(grad_mat_name, grad_dim_x, grad_dim_y)
        allocate_block.append(grad_malloc_str) 
    return allocate_block

def make_weights_init_block(ensembles_info):
    block = ["// initialize weights of layers "]
    for enm in ensembles_info:
        mat_name = enm[0]+"_weights"
        dim_x = enm[0]+".dim_x"
        dim_y = enm[0]+".dim_y"
        prev_dim_x = enm[0]+".prev_dim_x"
        prev_dim_y = enm[0]+".prev_dim_y"
        block.append(make_FC_weights_init(mat_name, dim_x, dim_y, prev_dim_x, prev_dim_y))
    return block
def make_weights_deallocate_block(ensembles_info):
    block = ["// deallocate weights of layers "]
    for enm in ensembles_info:
        mat_name = enm[0]+"_weights"
        block.append(make_FC_weights_free(mat_name))
    return block

def make_deallocate_block(ensembles_info):
    deallocate_block = ["// Deallocating memory for Output, Grad_output Matrices"]
    for enm in ensembles_info:
        deallocate_block.append(make_mkl_free(enm[0]+"_output")) 
        deallocate_block.append(make_mkl_free(enm[0]+"_grad_output")) 
    return deallocate_block

def make_loop_header(v, upper):
    return "for (int %s = 0; %s < %s; %s ++) " % (v, v, upper, v)

def make_init_solver(solver_info):
    assert solver_info is not None
    varname = solver_info["_name"].getChildren()[0]
    iterations = str(solver_info["_iter"].getChildren()[0])
    step_size = str(solver_info["_step"].getChildren()[0])
    print (varname, iterations, step_size)
    return "%s = SGD(%s, %s);" % (varname, iterations, step_size)

def make_solve_block(solver_info):
    solve_block = []
    iterations = str(solver_info["_iter"].getChildren()[0])
    solve_block.append(make_loop_header("iter", str(iterations))+"{")
    # TODO: load next instance of train data (feature and label)
    
    # TODO: forward propagation

    # TODO: annotate

    # TODO: backward propagation

    solve_block.append("}") # end the iteration loop
    return solve_block

ensembles_info = []
ensembles_info.append(("data_layer", 10, 10))
ensembles_info.append(("FC_layer", 20, 20))
ensembles_info.append(("loss_layer", 1, 1))

def main(program_file, cpp_file):
    # Front-end: processing program_file here
    py_compile.compile(program_file)
    AST = compiler.parseFile(program_file)  # get AST
    # managing info
    networks2enms = {}
    # pattern matching
    # (a) network
    patn_net = template_Network()
    matched = patn_net.matchall(AST)
    print "Network Matched: ", matched
    if matched:
        for net in patn_net.matches: 
            net_name = net["_name"].getChildren()[0]
            networks2enms.update({net_name:[]})

    # (b) Layers
    patn_fclayer = template_FullyConnectedLayer()
    matched = patn_fclayer.matchall(AST)
    print "FullyConnectedLayer Matched: ", matched
    if matched:
        for FClayer in patn_fclayer.matches: 
            print FClayer

    # (c) Solvers
    solver = None
    patn_solver = template_SGD()
    matched = patn_solver.matchall(AST)
    print "SGD Matched: ", matched
    if matched:
        for sgd in patn_solver.matches: 
            print sgd
            solver = sgd

    # CODE GENERATION:
    main_body_strs = []
    # allocating block 
    main_body_strs.append(make_allocate_block(ensembles_info))
    main_body_strs.append(make_weights_init_block(ensembles_info[1:]))

    # run solver
    main_body_strs.append([make_init_solver(solver)])
    main_body_strs.append(make_solve_block(solver))

    # deallocating block
    main_body_strs.append(make_weights_deallocate_block(ensembles_info[1:]))
    main_body_strs.append(make_deallocate_block(ensembles_info))

    # OUTPUT TO CPP FILE
    cpp_out = open(cpp_file, "w+")
    cpp_out.writelines([make_include_header(), make_newlines(2)])
    cpp_out.writelines([make_main_header(), make_newlines(2)])
    # TODO: output auxiliary function here
    for block in main_body_strs: 
        for statement in block:
            cpp_out.writelines([statement, make_newlines(1)])
        cpp_out.write(make_newlines(1))
    cpp_out.write("}") # ending bracket
    cpp_out.close()
    return

if __name__ == "__main__":
    if len(sys.argv) != NARGS:
        usage()
        sys.exit(1)
    else:
        program_file = sys.argv[1]
        cpp_file = sys.argv[2]
        main(program_file, cpp_file)
