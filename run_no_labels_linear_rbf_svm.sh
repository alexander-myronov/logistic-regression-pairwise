nice -n 9 python links_vs_npklr_vs_svm.py --file=data/benchmarks_svm_linear_adj_rand_2.csv \
--cv_folds=25 --rs_folds=4 --rs_iters=200 --jobs=65 --estimators_file=estimators_svm_linear.py \
--datasets_file=data/datasets_uci.txt

nice -n 9 python links_vs_npklr_vs_svm.py --file=data/benchmarks_svm_rbf_adj_rand_2.csv \
--cv_folds=25 --rs_folds=4 --rs_iters=200 --jobs=65 --estimators_file=estimators_svm_rbf.py \
--datasets_file=data/datasets_uci.txt

