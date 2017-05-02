# biclustering
The main function is the ssvd function, which implements sparse singular value decomposition. 
To install the package in Jupyter notebook, please use
"!pip install git+git://github.com/XingyuYan/biclustering.git".

Input variables: 
* X = n by d matrix
* gamu = weight parameter in adaptive lasso for the left singular vector
* gamv = weight parameter in adaptive lasso for the right singular vector
* merr = threshold to decide convergence
* niter = maximum number of iterations

Output variables:
* u = left sparse singular vector
* v = right sparse singular vector
* s = singular values
* iter = number of iterations to achieve convergence
