The command lines to generate the files are:

svm-train -t 0 ../examples/train model.1
svm-train -t 2 -g 0.5 ../examples/train model.4

./svm_classify.sh ../examples/test model.1 sys.1
./svm_classify.sh ../examples/test model.4 sys.4

