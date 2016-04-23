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

TODO LIST:
----------


- test sgemm\_axpy in Latte.cpp (code provided)

- write load\_train\_feature(int data\_index) in Latte.h and generate code
  for this part 

- parse annotate() for LossNeuron

- test the existing pattern matching and code generation framework with mlp.py

- develop more felexible and scalable framework of pattern matching

- loop tiling pass 

- loop fusion pass


Usage
-----

To generate code for given latte script, please run

    python generator.py [latte_script(.py)] [out_cpp_file(.cpp)]

Please put the generated cpp file "[out\_cpp\_file(.cpp)]" together with "Latte.h". 

To compile the generated code, enter the root of codebases (where you should place "Latte.h" and generated "[out\_cpp\_file(.cpp)]"), and run

    make 
    
Please make sure you have installed MKL library correctly and compile the generated cpp code using flags provided in the MKL compiling advisor. 



Assumptions
-----

- Assume only one data layer and loss layer in each network

- Assume Function name for data layer must contain "DataLayer"

- Assume Function name for loss layer must contain "LossLayer"

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

NOTES
---------
1. 
    


Acknowledgements
-------
We would like to give many thanks to Prof. Pingali and our Teaching Assistant 
Mr. Roshan for their teaching and instructions. 
