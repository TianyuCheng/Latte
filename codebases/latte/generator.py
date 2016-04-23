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
from optparse import OptionParser

from analyzer import * 
NARGS = 3

'''
def usage():
    """prints usage info when calling it"""
    usage_str = ""
    usage_str += "Usage: \n"
    usage_str += "\tpython [program_file(.py)] [out_cfile] \n"
    print usage_str
    return
'''

LATTE_H_PATH = '''"Latte.h"'''

def make_include_header():
    """makes the C++ header"""
    header = ""
    header += '''#include ''' + LATTE_H_PATH 
    header += '''\n\n//using namespace std;'''
    return header

def make_main_header():
    """makes the main header"""
    main_header = "int main (int argn, char** argv) { "
    return main_header

def make_newlines(num=1):
    """creates new lines based on argument passed it"""
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
    block = []
    for attr in attributes: 
        if allocate:
            block.append("// allocating memory for " + attr )
        else:
            block.append("// deallocating memory for " + attr )
        for enm in ensembles_info:
            _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
            output_mat_name = _cur+ "_" +attr
            if allocate:
                output_malloc_str = make_mkl_malloc(output_mat_name, _dim_x, 
                                    _dim_y, attributes[attr])
                block.append(output_malloc_str) 
            else:
                block.append(make_mkl_free(output_mat_name)) 
        #block.append("")
    # allocate for subtype of neuron
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y, _neurontype  = enm[:6]
        attributes = neuron_analyzers[_neurontype].fields
        if len(attributes) > 0:
            if allocate:
                block.append("// allocating memory for specific fields of " + _cur)
            else:
                block.append("// deallocating memory for specific fields of " + _cur)
        for attr in attributes: 
            output_mat_name = _cur+ "_" +attr
            if allocate:
                output_malloc_str = make_mkl_malloc(output_mat_name, _dim_x, _dim_y, attributes[attr])
                block.append(output_malloc_str) 
            else:
                block.append(make_mkl_free(output_mat_name)) 
        #block.append("")
    return block

def make_weights_init_block(ensembles_info, name2enm, allocate=True):
    if allocate:
        block = ["// initialize weights of layers "]
    else:
        block = ["// deallocate weights of layers "]
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        if "DataLayer" in _type: continue
        prev_dim_x = name2enm[_prev][3]
        prev_dim_y = name2enm[_prev][4]
        if allocate:
            init_str = "init_weights_mats(%s, %d, %d); " % (_cur+"_weights", prev_dim_x, prev_dim_y)
        else:
            init_str = make_FC_weights_free(_cur+"_weights")
        block.append(init_str)
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        if "DataLayer" in _type: continue
        prev_dim_x = name2enm[_prev][3]
        prev_dim_y = name2enm[_prev][4]
        if allocate:
            init_str = "init_weights_mats(%s, %d, %d); " % (_cur+"_grad_weights", prev_dim_x, prev_dim_y)
        else:
            init_str = make_FC_weights_free(_cur+"_grad_weights")
        block.append(init_str)   
    return block

def make_loop_header(v, start, upper, increment):
    """Creates a loop header (note there are no braces)"""
    return "for ( int %s = %s ; %s < %s ; %s = %s + %s ) " % \
           (v, start, v, upper, v, v, increment)

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


def make_solve_block(solver_info, ensembles_info, name2enm, bp_codes, fp_codes):
    solve_block = []
    iterations = str(solver_info["iter"])
    step_size = str(solver_info["step"])
    solve_block.append(make_loop_header("iter", 0, str(iterations), 1) + "{")
    solve_block.append("")


    solve_block.append(make_loop_header("data_idx", 0, "train_features.size()", 1) + "{")
    solve_block.append("")
    
    #  load next instance of train data (feature and label)
    load_label_str = "sgemm_copy (%s, train_features[data_idx], %s*%s);\n" % \
            (ensembles_info[0][0]+"_output", ensembles_info[0][3], ensembles_info[0][4])
    load_label_str += "vector<vector<int>> cur_label (%d, vector<int>(%d, 0));\n" % tuple(ensembles_info[-1][3:5])
    load_label_str += "cur_label[0][%s] = 1;" % "train_labels[data_idx]"
    solve_block.append(load_label_str)
    
    # forward propagation
    for enm in ensembles_info: 
        solve_block.append(fp_codes[enm[0]])
        solve_block.append("")

    # TODO: annotate

    # backward propagation
    for enm in ensembles_info[::-1]: 
        solve_block.append(bp_codes[enm[0]])
        solve_block.append("")
        
    # weights_update
    for enm in ensembles_info[1:]: 
        _cur, _type, _prev, _dim_x, _dim_y  = enm[:5]
        weights_update_str = "// weights_update for " + enm[0] + "\n"
        weights_update_str += "for (int x = 0; x < %s; x++) {\n" % _dim_x
        weights_update_str += "\tfor (int y = 0; y < %s; x++) {\n" % _dim_y
        weights_update_str += "\t\tsgemm_axpy(%s[x][y], %s, %s[x][y], %s*%s);\n" %\
                (_cur+"_weights", step_size, _cur+"_grad_weights", \
                    name2enm[_prev][3], name2enm[_prev][4])
        weights_update_str += "\t\tsgemm_zeros(%s[x][y], %s*%s);\n" % \
                (_cur+"_grad_weights", name2enm[_prev][3], name2enm[_prev][4])
        weights_update_str += "\t\tsgemm_zeros(%s[x][y], %s*%s);\n" % \
                (_cur+"_grad_output", name2enm[_prev][3], name2enm[_prev][4])
        weights_update_str += "\t}\n}"
        solve_block.append(weights_update_str)
    solve_block.append("")

    solve_block.append("} // end of data instances traversal") # end the train data sets loop
    solve_block.append("} // end of iterative traversal") # end the iteration loop
    return solve_block

def share_var_analyze (neuron_analyzers):
    # neuron analyzers is a dict that points a name to an analyzer
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

def main(options, program_file, cpp_file):
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
    # match all layers in AST
    for patn_layer in layer_templates:
        matched = patn_layer.matchall(AST)
        print patn_layer, "Matched: ", matched
        if matched:
            for layer in patn_layer.matches:
                # print layer
                net_name = layer['net']
                assert net_name in networks2enms
                if 'prev' not in layer: layer.update({"prev": None})
                
                layer['type'] = str(patn_layer).strip("template_")
                networks2enms[net_name].append(layer)
    print "###########################################"

    # (c) Solvers
    solver = None
    patn_solver = template_SGD()
    matched = patn_solver.matchall(AST)
    print "SGD Matched: ", matched
    if matched:
        for sgd in patn_solver.matches: 
            print sgd
            solver = sgd
    print "###########################################"

    #####################################################################
    # analyze lib functions and user-defined scripts
    # ensemble info is represented as a tuple
    ensembles_info = [ ( x['name'], \
                         x['type'], \
                         x['prev'], \
                         x['dim_x'], \
                         x['dim_y'], 
                         x['Neuron']) \
                      for x in networks2enms.values()[0] ]

    for x in ensembles_info: print x
    print "###########################################"

    # given an ensemble name, point it to the information tuple
    name2enm = {}
    for x in ensembles_info: name2enm.update({ x[0] : x })

    # create the neuron analyzers and also pass in ensemble info in order to create
    # forward and backward propogation code
    neuron_analyzers, fp_codes, bp_codes = \
            process_lib("lib.py", ensembles_info, name2enm, options.MKL_FLAG)
    for x in neuron_analyzers: print x, neuron_analyzers[x].fields

    #for x in fp_codes: print x, fp_codes[x]
    #for x in bp_codes: print x, fp_codes[x]
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
    main_body_strs.append(make_solve_block(solver, ensembles_info, name2enm, bp_codes, fp_codes))

    # deallocating block
    main_body_strs.append(make_weights_init_block(ensembles_info, name2enm, False))
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
    usage = "usage: python generator.py [options] arg1 arg2"
    parser = OptionParser(usage=usage)
    parser.add_option("-m", "--mkl", action="store_false", dest="MKL_FLAG", \
                      default=True, help="option to turn off pattern match for MKL calls.")
    parser.add_option("-t", "--tiling", action="store_false", dest="TILING_FLAG", \
                      default=True, help="option to turn off tiling optimization.")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="verbose")
    (options, args) = parser.parse_args()
    if len(args) != 2: 
        parser.print_help()
        parser.error("incorrect number of arguments")
    program_file = args[0]
    cpp_file = args[1]
    main(options, program_file, cpp_file)

