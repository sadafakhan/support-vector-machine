# support-vector-machine
```support-vector-machine``` implements a support vector machine created by libSVM to classify text documents. 

Args: 
* ```test_data```: data in the libSVM data format (cf. input/test)
* ```model_file```: model in the libSVM format (cf. examples/model_ex). The model file stores a_i*y_i for each support vector and p. 

Returns: 
* ```sys_output```: Each line in sys_output (cf. examples/sys_ex) has the format "trueLabel sysLabel fx"; trueLabel is the label in the gold standard, sysLable is the label produced by the SVM classifier, fx is the value of f(x) = wx - p = Sum(a_i * y_i * K(x_i, x) - p)

To run: 
```
src/svm_classify.sh input/test input/model.[X] output/sys.[X]
```

HW8 OF LING572 (02/28/2022)