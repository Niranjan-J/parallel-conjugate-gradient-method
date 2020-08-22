# Parallel Conjugate Gradient Method
Parallel implementation of an iterative algorithm for conjugate gradient method.\
CS 359: Parallel Computing Project

## Notes about our implementation

* We’ve created 2 libraries my_library.hpp (which contains main implementation) and my_testing_library.hpp (which contains methods for testing)


* We’ve created 3 driver programs -   
    1. solver.cpp (solves the problem given in source code)  
    2. solver-file.cpp (solves the problems specified in input.txt and writes the answers to output.txt)   
    3. tester.cpp (used for testing, generates random systems with different combinations of parameters and records the test results in a file)  
* For compiling tester.cpp: g++ tester.cpp -fopenmp -std=c++1z   
(requires c++ 17)  
* For others: g++ solver-file.cpp -fopenmp and g++ solver.cpp -fopenmp  
* Please refer to "Complete Test Results.xlsx" for test results

