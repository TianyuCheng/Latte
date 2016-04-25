Latte Project
=============

Authors
-------

- Student I
    + Name: Tianyu Cheng
    + EID: tc26752
    + Email: tianyu.cheng@utexas.edu

- Student II
    + Name: Loc Hoang
    + EID: ldh967
    + Email: loc@cs.utexas.edu

- Student III
    + Name: Xin Lin
    + EID: xl5224
    + Email: jimmylin@utexas.edu

Installation (Stampede)
-----------------------

To run this program, one must have intel MKL and boost installed. On TACC Stampede nodes, 
one needs to load intel icc and boost using the following commands:

```
module load intel/15.0.2
module load boost/1.55.0
```

After loading intel icc and boost, type:

```
make mlp
```

TODO LIST:
----------

- share variable analysis: determine uniform dependency of dimension; monitor data copy

- verbatim translation of forward/backward in SoA form (general form, no optimization)

- test the existing pattern matching and code generation framework with mlp.py

- develop more felexible and scalable framework of pattern matching

- loop tiling pass 

- loop fusion pass


Usage
-----

To generate code for given Latte script, please run

    python generator.py [latte_script(.py)] [out_cpp_file(.cpp)]

Please put the generated cpp file "[out\_cpp\_file(.cpp)]" together with "Latte.h". 

To compile the generated code, enter the root of codebases (where you should 
place "Latte.h" and generated "[out\_cpp\_file(.cpp)]"), and run the following
command:

    make 
    
Please make sure that you have installed the MKL library correctly and that you 
compile the generated cpp code using the flags provided in the MKL compiling 
advisor. 

System Details
-----

Latte is DSL/compiler system that attempt to make it easy for users to write
efficient neural network code using their defined DSL. The user must define
neurons, ensembles, and networks using the DSL, and the Latte compiler will
be able to create efficient neural network code.

Latte was implemented in Python in this system. 

# System Flow

* The user first has to write Python code in the Latte DSL format. The user
should define the neurons, layers, ensembles, etc. they wish to use and
specify how they are connected in the network that he or she wants to create.

* The system will then parse this code and create a Python AST from it.

* This Python AST is pattern-matched to some template, and from the pattern
matching the system generates C++ code. Some optimizations are done 
during this generation.

* The generated C++ code is analyzed 1 more time in order to produce 
a loop-tiled and loop-fused version of it.

* At this point, the system has created neural network code that can be
compiled using a regular compiler.


# Chosen File Summaries

## lib.py

This file comprises the core of the Python-Latte DSL. It defines classes such as
Neuron, Ensemble, and Network that are used by Latte to represent neural networks.

By using this file, one is able to write Python code using the Latte DSL model.

What follows are summaries of selected classes.

### Neuron

Represents a single neuron in a network. One must define forward and backward 
functions that compute forward and backward propogation for the neuron.

Note that a Neuron needs to be subtyped: it should not be used on its own.

### Ensemble

An ensemble is a (2-D) group of neurons. Neurons in an ensemble must all be of the
same type.

One connects ensembles together to create different kinds of layers in a network, 
such as Fully Connected Layers or Softmax Loss Layers.

### Network

A network represents a neural network, and it holds a set of ensembles that comprise
the neural network a user wishes to create. Note that the ensembles in question 
should already be connected to each other as the class does not hold connection 
information.

The class also tracks data sets and training data in order to run 
training/solving on a network.

## templates.py

This file defines a set of templates ASTs that will be used to match to
ASTs that are extracted from a Latte-Python description of a neural network.

Each template for has "wildcards" that are eventually replaced by data from a
the AST that each template is matching against.

TODO check if the following statement is right

By doing so ths system is able to have a some kind of uniformity when examining ASTs
since all ASTs will be matched to some template if they are valid.

## ast\_matcher.py

The system's AST matching implementation is in this file. 

Given an AST from some Latte-Python file, the system can use the match functions
defined in the file in order to attempt to match the AST to some
template defined in `templates.py`. The system also set the wildcards in the templates
by using data from the AST extracted from the Latte-Python file if a AST matches
on some template.

There are also definitions of AST visitor functions that do things such as count
binary operations, reorder binary operations, or rename nodes.

## analyzer.py

Neuron analyzer code is contained in this file.

A neuron analyzer is created for every type of neuron defined in the library:
in this case, `lib.py` is parsed in order to create an analyzer for all of
the neuron types contained in that file. 

An analyzer takes ensemble information and creates C++ code based on the
passed in ensemble information and the type of analyzer that the ensemble
information was passed into. Since forward/backward propogation information
is extracted from the library, the analyzer is able to create the forward
and backward propogation code for the ensemble. This code is then later
used to generate the neural network code.


## generator.py

This is the main program that generates code in the system. It defines many
functions that create C++ code for the system, and it is responsible for
putting the components of the system together.


As a review of what it does (and by extension a review of the system as well),
an AST is first parsed from a Latte-Python file. This AST is then pattern-matched
using the AST matcher capabilities of the system. From this the system
derives ensemble information, which is saved. The ensemble information is passed
into the neuron analyzers to create the forward and backward propogation code.
The code itself is then generated by creating code for all of the ensembles,
layers, solvers, and other things.

TODO not done yet since the system itself isn't complete yet; i.e. still
need to describe loop tiling and fusion pass


Assumptions
-----

- Assume only one data layer and loss layer in each network

- Assume the Function name for a data layer contains "DataLayer"

- Assume the Function name for a loss layer contains "LossLayer"

References
--------
We primarily referred to the Latte Paper provided in the course CS380C Compilers.


- [MKL Library] Documentation of Matrix Multiplication:

    https://software.intel.com/en-us/node/468480

- [MKL Library] Performance Comparison of Matrix Multiplication of MKL Library:

    http://www.ics.uci.edu/~paolo/FastMM/FMM-Reference/reference.html

- [MKL Library] Intel® Math Kernel Library Link Line Advisor:

    https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor

- [Python] API of Python AST Library (Abstract Syntax Tree): 

	https://docs.python.org/2/library/ast.html
	
- [Python] API of Python Parser Library:
	
	https://docs.python.org/2/library/parser.html

- Library of Libsvm Datasets:

   https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html

- [Other] Markdown

   https://guides.github.com/features/mastering-markdown/

NOTES
---------
1. 
    


Acknowledgements
-------
We would like to give many thanks to Prof. Pingali and our Teaching Assistant 
Mr. Roshan for their teaching and instructions. 
