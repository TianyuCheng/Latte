'''
    Latte Compiler 
'''

import os, sys
import compiler.ast
import inspect, compiler
from ast_matcher import *
from templates import *

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

def make_mkl_malloc(mat_name, dim_x, dim_y):
    return "double* %s = mkl_init_mat (%s, %s);" % (mat_name, dim_x, dim_y)

def make_mkl_free(mat_name): return "mkl_free(%s);" % (mat_name)

def make_FC_weights_init(name, dim_x, dim_y):
    return "vector<vector<double*>> %s (%s, %s);\n"

# input list of ensembles name
def make_allocate_block(ensembles_info):
    allocate_block = ["// Allocating memory for Output, Grad_output Matrices"]
    for enm in ensembles_info:
        output_mat_name = enm[0]+"_output"
        output_dim_x = enm[0]+".dim_x"
        output_dim_y = enm[0]+".dim_y"
        output_malloc_str = make_mkl_malloc(output_mat_name, output_dim_x, output_dim_y)
        allocate_block.append(output_malloc_str) 

        grad_mat_name = enm[0]+"_grad_output"
        grad_dim_x = enm[0]+".next_dim_x"
        grad_dim_y = enm[0]+".next_dim_y"
        grad_malloc_str = make_mkl_malloc(grad_mat_name, grad_dim_x, grad_dim_y)
        allocate_block.append(grad_malloc_str) 
    return allocate_block

def make_weight_init_block(ensembles_info):
    block = []
    for enm in ensembles_info:
        mat_name = enm[0]+"_weights"
        prev_dim_x = enm[0]+".prev_dim_x"
        prev_dim_y = enm[0]+".prev_dim_y"
        block.append(make_FC_weights_init(mat_name, prev_dim_x, prev_dim_y))
    return block

def make_deallocate_block(ensembles_info):
    deallocate_block = ["// Deallocating memory for Output, Grad_output Matrices"]
    for enm in ensembles_info:
        deallocate_block.append(make_mkl_free(enm[0]+"_output")) 
        deallocate_block.append(make_mkl_free(enm[0]+"_grad_output")) 
    return deallocate_block

def make_loop_header(v, upper):
    return "for (int %s = 0; %s < %s; %s ++) " % (v, v, upper, v)

def make_solve_block():
    solve_block = []
    solve_block.append(make_loop_header("iter", "ITERATIONS")+"{")
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
    # processing program_file here
    # get AST
    AST = compiler.parseFile(program_file)
    AST = ast_remove_module(AST)
    # pattern match
    patn_net = network()
    matched = patn_net.match(AST)
    print "Match?", matched
    if matched:
        for key, value in patn_net.wildcard.iteritems():
            print "%s:\t%s" % (key, value)
    ## OUTPUT: enm_names []

    main_body_strs = []
    # allocating block 
    main_body_strs.append(make_allocate_block(ensembles_info))

    # run solver
    main_body_strs.append(make_solve_block())

    # deallocating block
    main_body_strs.append(make_deallocate_block(ensembles_info))

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
