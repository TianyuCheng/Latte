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
    return "float* %s = init_mkl_mat (%s, %s);" % (mat_name, dim_x, dim_y)

def make_mkl_free(mat_name): return "mkl_free(%s);" % (mat_name)

def make_FC_weights_free(name):
    return "free_weights_mats(%s);" % (name)

# input list of ensembles name
def make_allocate_block(ensembles_info):
    allocate_block = ["// allocating memory for Output, Grad_output Matrices"]
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        output_mat_name = _cur+"_value"
        output_dim_x    = _dim_x
        output_dim_y    = _dim_y
        output_malloc_str = make_mkl_malloc(output_mat_name, output_dim_x, output_dim_y)
        allocate_block.append(output_malloc_str) 
    allocate_block.append("")
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        if "DataLayer" in _type: continue
        grad_mat_name = _cur+"_grad_value"
        grad_dim_x    = _dim_x
        grad_dim_y    = _dim_y
        grad_malloc_str = make_mkl_malloc(grad_mat_name, grad_dim_x, grad_dim_y)
        allocate_block.append(grad_malloc_str) 
    return allocate_block
def make_deallocate_block(ensembles_info):
    deallocate_block = ["// deallocating memory for Output, Grad_output Matrices"]
    for enm in ensembles_info:
        deallocate_block.append(make_mkl_free(enm[0]+"_output")) 
    for enm in ensembles_info:
        if "LossLayer" in enm[1]: continue
        deallocate_block.append(make_mkl_free(enm[0]+"_grad_output")) 
    return deallocate_block

def make_weights_init_block(ensembles_info, name2enm):
    block = ["// initialize weights of layers "]
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        if "DataLayer" in _type: continue
        declare_str = "vector<vector<float*>> %s (%s, vector<float*>(%s, NULL));" \
                % (_cur+"_weights", _dim_x, _dim_y)
        block.append(declare_str)
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        if "DataLayer" in _type: continue
        _prev = _prev.getChildren()[0]
        prev_dim_x = name2enm[_prev][3]
        prev_dim_y = name2enm[_prev][4]
        init_str = "init_weights_mats (%s, %d, %d);" % (_cur+"_weights", prev_dim_x, prev_dim_y)
        block.append(init_str)
    return block
def make_weights_deallocate_block(ensembles_info):
    block = ["// deallocate weights of layers "]
    for enm in ensembles_info:
        if "DataLayer" in enm[1]: continue
        mat_name = enm[0]+"_weights"
        block.append(make_FC_weights_free(mat_name))
    return block

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


def make_solve_block(solver_info, ensembles_info, name2enm):
    solve_block = []
    iterations = str(solver_info["iter"])
    solve_block.append(make_loop_header("iter", str(iterations))+"{")
    solve_block.append("")
    # TODO: load next instance of train data (feature and label)
    
    # TODO: forward propagation
    forward_str = "// Forward Propagation block \n"
    for enm in ensembles_info[1:]:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        _prev = _prev.getChildren()[0]
        # print _cur, _type, _prev, name2enm[_prev][3], name2enm[_prev][4]
        forward_str += "for (int i = 0; i < %s; i++) {\n" % (_dim_x)
        forward_str += "\tfor (int j = 0; j < %s; j ++) {\n" % (_dim_y)
        forward_str += "\t\tgemm(%s+j+i*%s, %s, %s[i][j], %s);\n" % \
                (_cur+"_value", _dim_y, _prev+"_value", _cur+"_weights", \
                  str(name2enm[_prev][3] * name2enm[_prev][4]))
        forward_str += "\t}\n}\n"
    solve_block.append(forward_str)

    # TODO: annotate

    # TODO: backward propagation

    solve_block.append("}") # end the iteration loop
    return solve_block

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
                if '_prev' not in layer: layer.update({"_prev": None})
                layer['_type'] = str(patn_layer).strip("template_")
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
    ensembles_info = [ ( x['_name'].getChildren()[0], \
                         x['_type'], \
                         x['_prev'], \
                         x['_dim_x'].getChildren()[0], \
                         x['_dim_y'].getChildren()[0]) \
                      for x in networks2enms.values()[0] ]
    name2enm = {}
    for x in ensembles_info: name2enm.update({ x[0] : x })
    print ensembles_info
    print name2enm
    main_body_strs.append(make_allocate_block(ensembles_info))
    main_body_strs.append(make_weights_init_block(ensembles_info, name2enm))

    # run solver
    #main_body_strs.append([make_init_solver(solver)])
    main_body_strs.append(make_solve_block(solver, ensembles_info, name2enm))

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
