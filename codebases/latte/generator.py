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

def make_mkl_malloc(mat_name, dim_x, dim_y, tp):
    return "%s %s = init_mkl_mat(%s, %s);" % (tp, mat_name, dim_x, dim_y)

def make_mkl_free(mat_name): return "mkl_free(%s);" % (mat_name)

def make_FC_weights_free(name):
    return "free_weights_mats(%s);" % (name)

# input list of ensembles name
def make_allocate_block(ensembles_info, neuron_analyzers, allocate=True):
    """ 
    allocate = True -->  Does allocation
    allocate = False --> Does deallocation
    """
    # allocate for base neuron type
    attributes = neuron_analyzers["Neuron"].fields
    allocate_block = []
    for attr in attributes: 
        allocate_block.append("// allocating memory for " + attr )
        for enm in ensembles_info:
            _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
            output_mat_name = _cur+ "_" +attr
            if allocate:
                output_malloc_str = make_mkl_malloc(output_mat_name, _dim_x, _dim_y, attributes[attr])
                allocate_block.append(output_malloc_str) 
            else:
                allocate_block.append(make_mkl_free(output_mat_name)) 
        #allocate_block.append("")
    # allocate for subtype of neuron
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y, _neurontype  = enm[:6]
        attributes = neuron_analyzers[_neurontype].fields
        if len(attributes) > 0:
            allocate_block.append("// allocating memory for specific fields of " + _cur)
        for attr in attributes: 
            output_mat_name = _cur+ "_" +attr
            if allocate:
                output_malloc_str = make_mkl_malloc(output_mat_name, _dim_x, _dim_y, attributes[attr])
                allocate_block.append(output_malloc_str) 
            else:
                allocate_block.append(make_mkl_free(output_mat_name)) 
        #allocate_block.append("")
    return allocate_block

def make_weights_init_block(ensembles_info, name2enm):
    block = ["// initialize weights of layers "]
    '''
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        if "DataLayer" in _type: continue
        declare_str = "vector<vector<float*>> %s(%s, vector<float*>(%s, NULL));" \
                % (_cur+"_weights", _dim_x, _dim_y)
        block.append(declare_str)
    '''
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        if "DataLayer" in _type: continue
        prev_dim_x = name2enm[_prev][3]
        prev_dim_y = name2enm[_prev][4]
        init_str = "init_weights_mats(%s, %d, %d); " % \
                (_cur+"_weights", prev_dim_x, prev_dim_y)
        block.append(init_str)
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        if "DataLayer" in _type: continue
        prev_dim_x = name2enm[_prev][3]
        prev_dim_y = name2enm[_prev][4]
        init_str = "init_weights_mats(%s, %d, %d);" % \
                (_cur+"_grad_weights", prev_dim_x, prev_dim_y)
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
            stmt_str = "Ensemble %s(%s, %s, %s); %s.add_ensemble(&%s);" % \
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
        # print _cur, _type, _prev, name2enm[_prev][3], name2enm[_prev][4]
        forward_str += "for (int x = 0; x < %d; x++) {\n" % (_dim_x)
        forward_str += "\tfor (int y = 0; y < %d; y ++) {\n" % (_dim_y)
        forward_str += "\t\tgemm(%s+y+x*%s, %s, %s[x][y], %s);\n" % \
                (_cur+"_value", _dim_y, _prev+"_value", _cur+"_weights", \
                  str(name2enm[_prev][3] * name2enm[_prev][4]))
        forward_str += "\t}\n}\n"
    solve_block.append(forward_str)

    # TODO: annotate

    # TODO: backward propagation


    solve_block.append("}") # end the iteration loop
    return solve_block

def share_var_analyze (neuron_analyzers):
    for neuron_name, neuron_analyzer in neuron_analyzers.iteritems():
        shared_vars = [ ]
        for field_name in neuron_analyzer.fields.iterkeys():
            # ignore all inputs
            if field_name.endswith("inputs"):
                shared_vars.append(field_name)
            if field_name.endswith("adj"):
                shared_vars.append(field_name)
        # delete all shared variables
        for shared_var in shared_vars:
            del neuron_analyzer.fields[shared_var]

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
                if 'prev' not in layer: layer.update({"prev": None})
                
                layer['type'] = str(patn_layer).strip("template_")
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

    #####################################################################
    # analyze lib functions and user-defined scripts
    neuron_analyzers = process_lib("lib.py")
    for x in neuron_analyzers:
        print x, neuron_analyzers[x].fields
    ensembles_info = [ ( x['name'], \
                         x['type'], \
                         x['prev'], \
                         x['dim_x'], \
                         x['dim_y'], 
                         x['Neuron']) \
                      for x in networks2enms.values()[0] ]
    for x in ensembles_info: print x
    name2enm = {}
    for x in ensembles_info: name2enm.update({ x[0] : x })
    share_var_analyze (neuron_analyzers)
    #####################################################################

    # CODE GENERATION:
    main_body_strs = []

    # creating network and ensembles
    main_body_strs.append(make_networks(networks2enms))
    main_body_strs.append(make_layers(networks2enms))
    
    # allocating block 
    main_body_strs.append(make_allocate_block(ensembles_info, neuron_analyzers))
    main_body_strs.append(make_weights_init_block(ensembles_info, name2enm))

    # run solver
    #main_body_strs.append([make_init_solver(solver)])
    main_body_strs.append(make_solve_block(solver, ensembles_info, name2enm))

    # deallocating block
    main_body_strs.append(make_weights_deallocate_block(ensembles_info))
    main_body_strs.append(make_allocate_block(ensembles_info, neuron_analyzers, False))

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
