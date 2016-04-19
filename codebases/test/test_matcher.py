#!/usr/bin/env python2

import os, sys
sys.path.append(os.path.abspath(".."))

import inspect, compiler
from latte.ast_matcher import *
from latte.templates import *

if __name__ == "__main__":
    tmpl = forloop()
    ast = compiler.parse("""
for i in range(len(inputs[0])):
    print inputs[0][i]
    """)
    ast = ast_remove_module(ast)
    matched = tmpl.match(ast)
    print "Match?", matched
    if matched:
        for key, value in tmpl.wildcard.iteritems():
            print "%s:\t%s" % (key, value)
