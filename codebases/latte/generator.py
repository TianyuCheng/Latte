#!/usr/bin/env python2
'''
    Latte Compiler 
'''

import os, sys
import ast
import inspect
from ast_matcher import *
from templates import *
import py_compile

from analyzer import *

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
    header += '''\n\n//using namespace std;'''
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
    declare_str = "vector<vector<double*>> %s (%s, vector<double*>(%s, NULL));\n" % (name, dim_x, dim_y)
    init_str = "init_weights_mats(%s, %s, %s);" % (name, prev_dim_x, prev_dim_y)
    return declare_str + init_str

def make_FC_weights_free(name):
    return "free_weights_mats(%s);" % (name)

# input list of ensembles name
def make_allocate_block(ensembles_info):
    allocate_block = ["// allocating memory for Output, Grad_output Matrices"]
    for enm in ensembles_info:
        output_mat_name = enm[0]+"_output"
        output_dim_x    = enm[0]+".dim_x"
        output_dim_y    = enm[0]+".dim_y"
        output_malloc_str = make_mkl_malloc(output_mat_name, output_dim_x, output_dim_y)
        allocate_block.append(output_malloc_str) 
    for enm in ensembles_info:
        if "LossLayer" in enm[1]: continue
        grad_mat_name = enm[0]+"_grad_output"
        grad_dim_x    = enm[0]+".next->dim_x"
        grad_dim_y    = enm[0]+".next->dim_y"
        grad_malloc_str = make_mkl_malloc(grad_mat_name, grad_dim_x, grad_dim_y)
        allocate_block.append(grad_malloc_str) 
    return allocate_block

def make_weights_init_block(ensembles_info):
    block = ["// initialize weights of layers "]
    for enm in ensembles_info:
        if "DataLayer" in enm[1]: continue
        mat_name = enm[0]+"_weights"
        dim_x = enm[0]+".dim_x"
        dim_y = enm[0]+".dim_y"
        prev_dim_x = enm[0]+".prev->dim_x"
        prev_dim_y = enm[0]+".prev->dim_y"
        block.append(make_FC_weights_init(mat_name, dim_x, dim_y, prev_dim_x, prev_dim_y))
    return block
def make_weights_deallocate_block(ensembles_info):
    block = ["// deallocate weights of layers "]
    for enm in ensembles_info:
        if "DataLayer" in enm[1]: continue
        mat_name = enm[0]+"_weights"
        block.append(make_FC_weights_free(mat_name))
    return block

def make_deallocate_block(ensembles_info):
    deallocate_block = ["// deallocating memory for Output, Grad_output Matrices"]
    for enm in ensembles_info:
        deallocate_block.append(make_mkl_free(enm[0]+"_output")) 
    for enm in ensembles_info:
        if "LossLayer" in enm[1]: continue
        deallocate_block.append(make_mkl_free(enm[0]+"_grad_output")) 
    return deallocate_block

def make_loop_header(v, upper):
    return "for (int %s = 0; %s < %s; %s ++) " % (v, v, upper, v)

def make_init_solver(solver_info):
    assert solver_info is not None
    varname = solver_info["name"]
    iterations = str(solver_info["iter"])
    step_size = str(solver_info["step"])
    print (varname, iterations, step_size)
    return "Solver %s = SGDSolver(%s, %s);" % (varname, iterations, step_size)

def make_networks(network_info):
    assert network_info is not None
    block = ["// create neural networks "]
    return block + [ "Network %s;" % net for net in network_info.iterkeys() ]

def make_layers(network_info):
    assert network_info is not None
    block = ["// create ensembles used in neural networks"]
    for net, ensembles in network_info.iteritems():
        for ensemble in ensembles:
            name = ensemble['name']
            dim_x = ensemble['dim_x']
            dim_y = ensemble['dim_y']
            if "DataLayer" in ensemble['type']: prev = "NULL"
            else: prev = "&" + ensemble['prev']
            net_name = ensemble['net']
            stmt_str = "Ensemble %s (%s, %s, %s); %s.add_ensemble(&%s);" % \
                    (name, dim_x, dim_y, prev, net_name, name)
            block.append(stmt_str)
    return block

def make_solve_block(solver_info):
    solve_block = []
    iterations = str(solver_info["iter"])
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
    AST = ast_parse_file(program_file)  # get AST
    # managing info
    networks2enms = {}

    # pattern matching
    # (a) network
    patn_net = template_Network()
    matched = patn_net.matchall(AST)
    print "Network Matched: ", matched
    if matched:
        for net in patn_net.matches: 
            net_name = net["name"]
            networks2enms.update({net_name:[]})

    # (b) Layers
    for patn_layer in layer_templates:
        matched = patn_layer.matchall(AST)
        print patn_layer, "Matched: ", matched
        if matched:
            for layer in patn_layer.matches:
                print layer
                net_name = layer['net']
                assert net_name in networks2enms
                layer['type'] = str(patn_layer)
                networks2enms[net_name].append(layer)

    # (c) Solvers
    solver = None
    patn_solver = template_SGD()
    matched = patn_solver.matchall(AST)
    print "SGD Matched: ", matched
    if matched:
        for sgd in patn_solver.matches: 
            print sgd
            solver = sgd

    # analyze lib functions and user-defined scripts
    process_lib("lib.py")
    # process_lib("../test/test_dsl.py")

    # CODE GENERATION:
    main_body_strs = []

    # creating network and ensembles
    main_body_strs.append(make_networks(networks2enms))
    main_body_strs.append(make_layers(networks2enms))
    
    # allocating block 
    ensembles_info = [ ( x['name'], \
                         x['type'] )  \
                      for x in networks2enms.values()[0] ]
    print ensembles_info
    main_body_strs.append(make_allocate_block(ensembles_info))
    main_body_strs.append(make_weights_init_block(ensembles_info))

    # run solver
    #main_body_strs.append([make_init_solver(solver)])
    main_body_strs.append(make_solve_block(solver))

    # deallocating block
    main_body_strs.append(make_weights_deallocate_block(ensembles_info))
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
