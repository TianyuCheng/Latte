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
from term import *

from analyzer import * 
NARGS = 3

from optimizer import TilingOptimizer

'''
def usage():
    """prints usage info when calling it"""
    usage_str = ""
    usage_str += "Usage: \n"
    usage_str += "\tpython [program_file(.py)] [out_cfile] \n"
    print usage_str
    return
'''

# cout << "iter: " << iter << ", data_idx: " << data_idx << endl;
LATTE_H_PATH = '''"Latte.h"'''

def make_define_header(options, networks2enms):
    defines = []
    for net, enms in networks2enms.iteritems():
        for enm in enms:
            if "pool_dim_x" in enm:
                defines.append("#define %s_pool_dim_x   (%d)" % (enm['name'], enm['pool_dim_x']))
            if "pool_dim_y" in enm:
                defines.append("#define %s_pool_dim_y   (%d)" % (enm['name'], enm['pool_dim_y']))
    return '\n'.join(defines)

def make_include_header(options):
    """makes the C++ header"""
    header = ""
    header += '''#include ''' + LATTE_H_PATH 
    batch_parallel_flag = options.DP_FLAG
    tiling_flag = options.TILING_FLAG
    if batch_parallel_flag or tiling_flag: 
        header += '''\n#include <omp.h>'''
    header += '''\n\n//using namespace std;'''
    return header

def make_main_header():
    """makes the main header"""
    main_header = "int main (int argn, char** argv) { "
    return main_header

def make_omp_init_block(options):
    numWorkers = options.NWORKERS
    batch_parallel_flag = options.DP_FLAG
    tiling_flag = options.TILING_FLAG
    if not (batch_parallel_flag or tiling_flag): return []
    block = [ "// OMP Library Initialization Block" ]
    omp_init_str = "omp_set_num_threads(%d);" % numWorkers
    block.append(omp_init_str)
    return block

def make_newlines(num=1):
    """creates new lines based on argument passed it"""
    return "\n" * num

# def make_indent(num=indent):
#     return "    " * num

def make_mkl_malloc(options, mat_name, dim_x, dim_y, tp, share=False):
    if tp == "float*":
        if options.DP_FLAG and not share: 
            string = "vector<%s> %s (%d, NULL); \n" % (tp, mat_name, options.NWORKERS)
            string += "for (int i = 0; i < %d; i ++) %s[i] = init_mkl_mat(%s, %s);" \
                    % (options.NWORKERS, mat_name, dim_x, dim_y)
            return string
        else: return "%s %s = init_mkl_mat(%s, %s);" % (tp, mat_name, dim_x, dim_y)
    elif tp == "vector<vector<float*>>":
        if options.DP_FLAG and "grad_weights" in mat_name:
            return "vector<%s> %s (%d, vector<vector<float*>>(%s, vector<float*>(%s, NULL)));" % \
                        (tp, mat_name, options.NWORKERS, dim_x, dim_y)
        else:
            return "%s %s (%s, vector<float*>(%s, NULL));" % (tp, mat_name, dim_x, dim_y)
        

def make_mkl_free(options, mat_name, tp): 
    if tp == "float*":
        if options.DP_FLAG: 
            return "for (int i = 0; i < %d; i++) mkl_free(%s[i]);" % \
                    (options.NWORKERS, mat_name)
        else: return "mkl_free(%s);" % (mat_name)
    elif tp == "vector<vector<float*>>":
        if options.DP_FLAG and "grad_weights" in mat_name:
            return "for (int i = 0; i < %d; i++) free_weights_mats(%s[i]);" % \
                    (options.NWORKERS, mat_name)
        else: return "free_weights_mats(%s);" % (mat_name)

def make_FC_weights_free(name):
    return "free_weights_mats(%s);" % (name)

# input list of ensembles name
def make_allocate_block(options, ensembles_info, neuron_analyzers, conn_types, allocate=True):
    """ 
    allocate = True -->  Does allocation
    allocate = False --> Does deallocation
    """
    block = []
    # allocate for subtype of neuron
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y, _neurontype, _aux  = enm[:7]
        attributes = neuron_analyzers[_neurontype].fields
        if allocate: print _cur, _type, attributes
        if len(attributes) > 0:
            if allocate:
                block.append("// allocating memory for specific fields of " + _cur)
            else:
                block.append("// deallocating memory for specific fields of " + _cur)
        for attr in attributes: 
            mat_name = _cur+ "_" +attr
            
            if allocate:
                if _type in conn_types and conn_types[_type][2]:
                    if (attr == "weights" or attr == "grad_weights"):
                        attributes[attr] = "float*"
                        block.append(make_mkl_malloc(options, mat_name, \
                                         _aux['ker_dim_x'], _aux['ker_dim_y'], attributes[attr], True)) 
                        continue
                block.append(make_mkl_malloc(options, mat_name, _dim_x, _dim_y, attributes[attr])) 
            else:
                block.append(make_mkl_free(options, mat_name, attributes[attr])) 
        #block.append("")
    return block

def make_weights_init_block(options, ensembles_info, name2enm, conn_types, allocate=True):
    ''' Weights are shared by all threads, so nothing to do with data parallelism '''
    if allocate:
        block = ["// initialize weights of layers "]
    else:
        block = ["// deallocate weights of layers "]
    ### for "weights"
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y, _neurontype  = enm[:6]
        attributes = neuron_analyzers[_neurontype].fields
        if "DataLayer" in _type: continue
        if "weights" not in attributes: continue
        prev_dim_x = name2enm[_prev][3]
        prev_dim_y = name2enm[_prev][4]
        if allocate:
            mat_name = _cur + "_weights"
            if _type in conn_types and conn_types[_type][2]:
                n_prev = str(prev_dim_x) + " * " + str(prev_dim_y)
                n_cur = str(_dim_x) + " * " + str(_dim_y)
                init_str = "Xaiver_initialize(%s, %s, %s);" % (mat_name, n_prev, n_cur)
            else:
                init_str = "init_weights_mats(%s, %d, %d, true); " % (mat_name, prev_dim_x, prev_dim_y)
        else:
            init_str = make_FC_weights_free(_cur+"_weights")
        block.append(init_str)
    ### for "grad_weights"
    for enm in ensembles_info:
        _cur, _type, _prev, _dim_x, _dim_y, _neurontype  = enm[:6]
        attributes = neuron_analyzers[_neurontype].fields
        if "DataLayer" in _type: continue
        if "grad_weights" not in attributes: continue
        prev_dim_x = name2enm[_prev][3]
        prev_dim_y = name2enm[_prev][4]
        if allocate:
            init_str, subscript = "", ""
            if options.DP_FLAG:
                init_str += "for (int i = 0; i < %d; i ++) " % options.NWORKERS
                subscript += "[i]"
            mat_name = _cur + "_grad_weights"
            if _type in conn_types and conn_types[_type][2]: pass
            else: 
                init_str += "init_weights_mats(%s, %d, %d, false); " \
                        % (_cur+"_grad_weights"+subscript, prev_dim_x, prev_dim_y)
        else:
            init_str = make_FC_weights_free(_cur+"_grad_weights")
        block.append(init_str)   
    return block

def make_load_data(options, networks2enms):
    for net, ensembles in networks2enms.iteritems():
        enm = ensembles[0]
        load_block = []
        if enm["type"] == "LibsvmDataLayer": data_format = "libsvm"
        elif enm["type"] == "MnistDataLayer": data_format = "mnist"

        load_block.append("// load " + data_format + " data")
        load_block.append("vector<float*> train_features, test_features;");
        load_block.append("vector<int> train_labels, test_labels;");

        sub = ""
        if options.mini:
            sub = ".mini"

        # we need number of features
        load_block.append("""read_%s("%s", train_features, train_labels, %d, %d, %d);""" % (\
            data_format, enm["train_file"] + sub, enm["dim_x"], enm["dim_y"], enm['nLabels']))
        load_block.append("""read_%s("%s", test_features, test_labels, %d, %d, %d);""" % (\
            data_format, enm["test_file"], enm["dim_x"], enm["dim_y"], enm['nLabels']))
        
        load_block.append("assert (train_features.size() == train_labels.size());")
        load_block.append("assert (test_features.size() == test_labels.size());")
        load_block.append("vector<int> shuffle_index;" )
        load_block.append("generate_shuffle_index(shuffle_index, train_features.size());" )
    return load_block

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

def make_test_block(solver_info, ensembles_info, name2enm, fp_codes,
                    forwards_ensemble_order, backwards_ensemble_order):
    test_block = []
    test_block.append("// test block")
    test_block.append("vector<int> preds;")
    test_block.append(make_loop_header("data_idx", 0, "test_features.size()", 1) + "{")
    #test_block.append("float dp_result;")
    test_block.append("int tid = 0;")
    test_block.append("")

    if options.DP_FLAG: subscript = "[tid]"
    else: subscript = ""
    # load next instance of train data (feature and label)
    load_label_str = "sgemm_copy (%s, test_features[data_idx], %s*%s);\n" % \
            (ensembles_info[0][0]+"_output"+subscript, ensembles_info[0][3], ensembles_info[0][4])
    load_label_str += "vector<vector<int>> cur_label (%d, vector<int>(%d, 0));\n" % tuple(ensembles_info[-1][3:5])
    load_label_str += "cur_label[0][%s] = 1;" % "test_labels[data_idx]"
    test_block.append(load_label_str)
    
    # forward propagation
    for i in forwards_ensemble_order: 
        if fp_codes[i] == None:
            continue

        test_block.append(str(fp_codes[i]))
        test_block.append("")

    # annotate
    _cur, _type, _prev, _dim_x, _dim_y  = ensembles_info[-1][:5]
    loss_layer_output = _cur+"_output"
    annotate_str = "// annotate for loss layer in testing stage\n"
    annotate_str += "int pred = argmax (%s, %s*%s);\n" % \
            (loss_layer_output+subscript, _dim_x, _dim_y)
    annotate_str += "preds.push_back(pred);\n"
    test_block.append(annotate_str)

    test_block.append("}")
    
    # evaluation
    eval_str = "// evaluate the accuracy performance\n"
    eval_str += "evaluate(preds, test_labels);"
    test_block.append(eval_str)

    return test_block

def make_solve_block(options, conn_types, neuron_analyzers, solver_info, ensembles_info, name2enm, bp_codes, 
                     fp_codes, forwards_ensemble_order, backwards_ensemble_order):
    solve_block = []

    # store time for each iteration
    solve_block.append("vector<float> times;")
    solve_block.append("timespec start;")
    solve_block.append("timespec stop;")

    iterations = str(solver_info["iter"])
    if solver_info["step"] > 0: step_size = str(solver_info["step"] * -1.0)
    solve_block.append("// solve block")
    # solve_block.append(make_loop_header("iter", 0, str(iterations), 1) + "{")
    solve_block.append(make_loop_header("iter", 0, 1, 1) + "{")
    solve_block.append("")

    # measure iteration time time
    solve_block.append("clock_gettime(CLOCK_MONOTONIC, &start);");

    # Data parallel: add pragma directive here (a new nested loop with batch)
    numWorkers = options.NWORKERS
    batch_parallel_flag = options.DP_FLAG
    tiling_flag = options.TILING_FLAG
    if batch_parallel_flag: 
        omp_directive_str = "#pragma omp parallel for"
        #if tiling_flag: omp_directive_str += " collapse(2)"
        omp_directive_str += " schedule(static, 2)"
        #omp_directive_str += " private(tid, data_idx, cur_label, sumover)"
        solve_block.append(omp_directive_str)
    solve_block.append(make_loop_header("si", 0, "train_features.size()", 1) + "{")
    if batch_parallel_flag:
        solve_block.append("int tid = omp_get_thread_num();")
    solve_block.append("")
    
    #  load next instance of train data (feature and label)
    load_label_str = "int data_idx = shuffle_index[si];\n"
    if options.DP_FLAG: subscript = "[tid]"
    else: subscript = ""
    load_label_str += "sgemm_copy (%s, train_features[data_idx], %s*%s);\n" % \
            (ensembles_info[0][0]+"_output"+subscript, ensembles_info[0][3], ensembles_info[0][4])
    load_label_str += "vector<vector<int>> cur_label (%d, vector<int>(%d, 0));\n" % tuple(ensembles_info[-1][3:5])
    load_label_str += "cur_label[0][%s] = 1;\n" % "train_labels[data_idx]"
    #load_label_str += "float dp_result;"
    solve_block.append(load_label_str)
    
    # forward propagation
    for i in forwards_ensemble_order: 
        if fp_codes[i] == None:
            continue

        solve_block.append(str(fp_codes[i]))
        solve_block.append("")

    # annotate
    _cur, _type, _prev, _dim_x, _dim_y  = ensembles_info[-1][:5]
    annotate_str = "// annotate for loss layer\n"
    annotate_str += "float sumover = 0.0;\n"
    annotate_str += "for (int x = 0; x < %s; x++) {\n" % _dim_x
    annotate_str += "\tfor (int y = 0; y < %s; y++) {\n" % _dim_y
    annotate_str += "\t\tsumover += *(%s+x*%s+y);\n" % (_cur+"_output"+subscript, _dim_y)
    annotate_str += "\t}\n}\n"
    annotate_str += "for (int x = 0; x < %s; x++) {\n" % _dim_x
    annotate_str += "\tfor (int y = 0; y < %s; y++) {\n" % _dim_y
    annotate_str += "\t\t*(%s+x*%s+y) = *(%s+x*%s+y) / sumover;\n" % \
            (_cur+"_output"+subscript, _dim_y, _cur+"_output"+subscript, _dim_y)
    annotate_str += "\t}\n}\n"
    solve_block.append(annotate_str)

    # backward propagation
    for i in backwards_ensemble_order:
        if bp_codes[i] == None:
            continue
        solve_block.append(str(bp_codes[i]))
        solve_block.append("")
        
    # weights_update
    for enm in ensembles_info[1:]: 
        _cur, _type, _prev, _dim_x, _dim_y, _neurontype, _aux  = enm[:7]
        attributes = neuron_analyzers[_neurontype].fields
        prev_dim_x, prev_dim_y = name2enm[_prev][3], name2enm[_prev][4]
        is_shared_weights = _type in conn_types and conn_types[_type][2]

        weights_update_str = "// weights_update for " + enm[0] + "\n"
        if is_shared_weights:
            ker_dim_x, ker_dim_y = _aux['ker_dim_x'], _aux['ker_dim_y']
            weights_update_str += "\t\tsgemm_axpy(%s, %s, %s, %s*%s);\n" %\
                    (_cur+"_weights", step_size, _cur+"_grad_weights", \
                         ker_dim_x, ker_dim_y)
            weights_update_str += "\t\tsgemm_zeros(%s, %s*%s);\n" % \
                        (_cur+"_grad_weights", ker_dim_x, ker_dim_y)
        if "weights" in attributes and not is_shared_weights: 
            weights_update_str += "for (int x = 0; x < %s; x++) {\n" % _dim_x
            weights_update_str += "\tfor (int y = 0; y < %s; y++) {\n" % _dim_y
            if options.DP_FLAG: 
                weights_update_str += "\t\tfor (int i = 0; i < %s ; i ++) {\n" % prev_dim_x
                weights_update_str += "\t\tfor (int j = 0; j < %s ; j ++) {\n" % prev_dim_y
                weights_update_str += "#pragma omp atomic\n"
                weights_update_str += \
                        "*(%s[x][y]+i*%s+j) = *(%s[x][y]+i*%s+j) + (%s) * (*(%s[tid][x][y]+i*%d+j));\n" \
                        % (_cur+"_weights", prev_dim_y, _cur+"_weights", prev_dim_y, \
                           step_size, _cur+"_grad_weights", prev_dim_y) 
                weights_update_str += "\t\t}\n\t\t}\n"
                subscript = "[tid]"
            else: 
                subscript = ""
                weights_update_str += "\t\tsgemm_axpy(%s[x][y], %s, %s[x][y], %s*%s);\n" %\
                        (_cur+"_weights", step_size, _cur+"_grad_weights"+subscript, \
                         prev_dim_x, prev_dim_y)
            weights_update_str += "\t\tsgemm_zeros(%s[x][y], %s*%s);\n" % \
                        (_cur+"_grad_weights"+subscript, name2enm[_prev][3], name2enm[_prev][4])
            weights_update_str += "\t}\n}\n"

        if "grad_output" in attributes:
            weights_update_str += "\t\tsgemm_zeros(%s, %s*%s);\n" % \
                    (_cur+"_grad_output"+subscript, _dim_x, _dim_y)
        solve_block.append(weights_update_str)
    solve_block.append("")

    solve_block.append("} // end of data instances traversal") # end the train data sets loop

    # time measurement
    solve_block.append("clock_gettime(CLOCK_MONOTONIC, &stop);");
    solve_block.append("timespec t = time_diff(start, stop);");
    solve_block.append("times.push_back(t.tv_sec);");
    solve_block.append('cout << "time for iter(s): " << t.tv_sec << endl;');

    solve_block.append("} // end of iterative traversal") # end the iteration loop
    return solve_block

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
                layer['type'] = str(patn_layer).strip("template_")

                if 'prev' not in layer: layer.update({"prev": None})
                if 'Neuron' not in layer: 
                    if layer['type'] == 'LibsvmDataLayer':
                        layer['Neuron'] = 'DataNeuron'
                    elif layer['type'] == 'MnistDataLayer':
                        layer['Neuron'] = 'DataNeuron'
                    elif layer['type'] == 'SoftmaxLossLayer':
                        layer['Neuron'] = 'SoftmaxNeuron'
                    else:
                        # print "##", layer
                        assert False
                networks2enms[net_name].append(layer)
    print "###########################################"
    # put the layers in correct order by looking for their previous layer
    for net_name in networks2enms.iterkeys():
        layer_names = map(lambda x: x['name'], networks2enms[net_name])
        layer_dict = dict(zip(layer_names, networks2enms[net_name]))
        layers = filter(lambda x: x['prev'] == None, networks2enms[net_name])
        if len(layers) != 1:
            print "ERROR (NEXT LAYER): ", layers
        assert len(layers) == 1
        layer_name = layers[0]['name']
        num_layers = len(networks2enms[net_name]) - 1
        while num_layers > 0:
            next_layer = filter(lambda x: x['prev'] == layer_name, networks2enms[net_name])
            print next_layer
            if len(next_layer) != 1:
                print "ERROR (NEXT LAYER): ", next_layer
            assert len(next_layer) == 1
            num_layers -= 1
            layers = layers + next_layer
            layer_name = next_layer[0]['name']
        networks2enms[net_name] = layers
        print "Network %s:" % net_name, map(lambda x: x['name'], layers)

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
                         x['Neuron'], x) \
                      for x in networks2enms.values()[0] ]

    for x in ensembles_info: print x
    print "###########################################"

    # given an ensemble name, point it to the information tuple
    name2enm = {}
    for x in ensembles_info: name2enm.update({ x[0] : x })

    path = os.path.dirname(os.path.abspath(__file__))
    # parse the add_connection calls in stdlib
    # and perform the shared variable analysis
    conn_types = process_add_connection(path + "/lib.py", name2enm)
    for net, ensembles in networks2enms.iteritems():
        for ensemble in ensembles:
            layer_type = ensemble['type']
            if layer_type in conn_types:
                args, mapping, _ = conn_types[layer_type]
                term.dump("Layer %s uniform dependency? %s" % (layer_type, \
                        check_uniform_dependency(args, mapping, ensemble, name2enm)), \
                        term.OKBLUE)

    # create the neuron analyzers and also pass in ensemble info in order to create
    # forward and backward propogation code
    neuron_analyzers, fp_codes, bp_codes, fp_code_list, bp_code_list = \
            process_lib(path + "/lib.py", ensembles_info, name2enm, conn_types, options)
    # for x in neuron_analyzers: print x, neuron_analyzers[x].fields

    #for x in fp_codes: print x, fp_codes[x]
    #for x in bp_codes: print x, fp_codes[x]
    # share_var_analyze (neuron_analyzers)
    #####################################################################
    # print "#####################################33"
    # for key, value in fp_codes.iteritems():
    #     print key, type(value)
    # print "#####################################33"

    
    forwards_ensemble_order = []
    for i in ensembles_info:
        forwards_ensemble_order.append(i[0])
        
    backwards_ensemble_order = []
    for i in ensembles_info[::-1]:
        backwards_ensemble_order.append(i[0])

    tiling_flag = options.TILING_FLAG

    # if tiling flag is set, then run the tiling
    tiling_flag = options.TILING_FLAG
    if tiling_flag:
        # forward
        opt1 = TilingOptimizer(fp_codes, forwards_ensemble_order)
        forwards_ensemble_order = opt1.optimize()

        # backward
        opt2 = TilingOptimizer(bp_codes, backwards_ensemble_order)
        backwards_ensemble_order = opt2.optimize()

    # CODE GENERATION:
    main_body_strs = []

    # OMP initialization block
    main_body_strs.append(make_omp_init_block(options))

    # creating network and ensembles
    main_body_strs.append(make_networks(networks2enms))
    main_body_strs.append(make_layers(networks2enms))
    
    # allocating block 
    main_body_strs.append(make_allocate_block(options, ensembles_info, \
                                              neuron_analyzers, conn_types))
    main_body_strs.append(make_weights_init_block(options, ensembles_info, name2enm, conn_types))

    # load data
    main_body_strs.append(make_load_data(options, networks2enms))
    main_body_strs.append(['cout << "Loaded Data Successfully" << endl;'])

    # run solver
    #main_body_strs.append([make_init_solver(solver)])
    main_body_strs.append(make_solve_block(options, conn_types, neuron_analyzers, solver, ensembles_info, 
                          name2enm, bp_codes, fp_codes, forwards_ensemble_order,
                          backwards_ensemble_order))

    # run tester
    main_body_strs.append(make_test_block(solver, ensembles_info, name2enm, fp_codes,
                          forwards_ensemble_order, backwards_ensemble_order))

    # # deallocating block
    # #main_body_strs.append(make_weights_init_block(ensembles_info, name2enm, False))
    # main_body_strs.append(make_allocate_block(options, ensembles_info, \
    #                       neuron_analyzers, conn_types, False))

    # OUTPUT TO CPP FILE
    cpp_out = open(cpp_file, "w+")
    cpp_out.writelines([make_include_header(options), make_newlines(2)])
    cpp_out.writelines([make_define_header(options, networks2enms), make_newlines(2)])
    cpp_out.writelines([make_main_header(), make_newlines(2)])
    # TODO: output auxiliary function here
    for block in main_body_strs: 
        for statement in block:
            if statement is not None:
                cpp_out.writelines([str(statement), make_newlines(1)])
        cpp_out.write(make_newlines(1))
    cpp_out.write("}") # ending bracket
    cpp_out.close()
    return

if __name__ == "__main__":
    usage = "usage: python generator.py [options] py_script cpp_out_file"
    parser = OptionParser(usage=usage)
    parser.add_option("-m", "--mkl", action="store_true", dest="MKL_FLAG", \
                      default=False, help="option to turn on pattern match for MKL calls.")
    parser.add_option("-b", "--batch-parallel", action="store_true", dest="DP_FLAG", \
                      default=False, help="option to turn on batch parallelization.")
    parser.add_option("-t", "--tiling-parallel", action="store_true", dest="TILING_FLAG", \
                      default=False, help="option to turn on loop tiling.")
    parser.add_option("-w", "--numWorkers", action="store", type="int", dest="NWORKERS", \
                      default=1, help="Specify the allocated number of threads \
                      for parallel computing (needed when -b or -t is on)")
    parser.add_option("-f", "--fusion", action="store_true", dest="FUSION_FLAG", \
                      default=False, help="option to turn on fusion functionality.")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", help="verbose")
    parser.add_option("", "--mini", action="store_true", dest="mini", help="mini")
    (options, args) = parser.parse_args()
    if len(args) != 2: 
        parser.print_help()
        parser.error("incorrect number of arguments")
    program_file = args[0]
    cpp_file = args[1]
    main(options, program_file, cpp_file)
