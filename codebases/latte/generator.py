'''
    Latte Compiler 
'''

import os, sys


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
    header += '''#include ''' + LATTE_H_PATH + "\n"
    return header

def make_main_header():
    main_header = "int main(int argn, char** argv) { \n"
    return main_header

def make_newlines(num=1):
    return "\n" * num


def main(program_file, cpp_file):
    # processing program_file here
    py_in = open(program_file, "r")
    # get AST
    # pattern match

    main_body_strs = []
    py_in.close()

    cpp_out = open(cpp_file, "w+")
    cpp_out.writelines([make_include_header(), make_newlines(1)])
    # TODO: output auxiliary function here
    for block in main_body_strs: 
        cpp_out.writelines([block, make_newlines(1)])
    cpp_out.writelines([make_main_header(), make_newlines(1)])
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
