# Semi-supervised naive Bayes

The Python module in this repository implements the semi-supervised version of naive Bayes described in Section 5.3.1 of the following paper:

K. Nigam, A.K. McCallum, S. Thrun, T. Mitchell (2000). Text classification from labeled and unlabeled documents using EM. Machine learning 39(2-3), pp. 103-134.

The code is a modified version of the scikit-learn multinomial naive Bayes classifier.

# Requirements

The module requires NumPy, SciPy, and scikit-learn.  It has been tested on NumPy 1.18.1, SciPy 1.4.1, scikit-learn 0.22.1, and Python 3.7.

# Usage

The `MultinomialNBSS` class implements several of the common scikit-learn classifier methods, such as `fit` and `predict`.  When calling the `fit` method, all unlabeled data should be marked with a `-1`.