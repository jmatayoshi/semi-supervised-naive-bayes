# The code in this module is adapted from scikit-learn's Naive Bayes classifier.
# The original code has been altered to implement the semi-supervised version of
# Naive Bayes described in Section 5.3.1 of the following paper:

# K. Nigam, A.K. McCallum, S. Thrun, T. Mitchell (2000). Text classification
# from labeled and unlabeled documents using EM. Machine Learning 39(2-3),
# pp. 103-134.

#
# Original copyright notice below:
# Author: Vincent Michel <vincent.michel@inria.fr>
#         Minor fixes by Fabian Pedregosa
#         Amit Aides <amitibo@tx.technion.ac.il>
#         Yehuda Finkelstein <yehudaf@tx.technion.ac.il>
#         Lars Buitinck
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#         (parts based on earlier work by Mathieu Blondel)
#
# License: BSD 3 clause

import numpy as np
from scipy.sparse import issparse

from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

class MultinomialNBSS(_BaseDiscreteNB):
    """
    Semi-supervised Naive Bayes classifier for multinomial models.  Unlabeled
    data must be marked with -1.  In comparison to the standard scikit-learn
    MultinomialNB classifier, the main differences are in the _count and fit
    methods.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    beta : float, optional (default=1.0)
        Weight applied to the contribution of the unlabeled data
        (0 for no contribution)

    fit_prior : boolean, optional (default=True)
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like, size (n_classes,), optional (default=None)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    tol : float, optional (default=1e-3)
        Tolerance for convergence of EM algorithm.

    max_iter : int, optional (default=20)
        Maximum number of iterations for EM algorithm.

    verbose : boolean, optional (default=True)
        Whether to ouput updates during the running of the EM algorithm.

    Attributes
    ----------
    class_log_prior_ : array, shape (n_classes, )
        Smoothed empirical log probability for each class.

    intercept_ : array, shape (n_classes, )
        Mirrors ``class_log_prior_`` for interpreting MultinomialNBSS
        as a linear model.

    feature_log_prob_ : array, shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.

    coef_ : array, shape (n_classes, n_features)
        Mirrors ``feature_log_prob_`` for interpreting MultinomialNBSS
        as a linear model.

    class_count_ : array, shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : array, shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from semi_supervised_naive_bayes import MultinomialNBSS
    >>> clf = MultinomialNBSS()
    >>> clf.fit(X, y)
    MultinomialNBSS(alpha=1.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2:3]))
    [3]

    Notes
    -----
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html

    K. Nigam, A.K. McCallum, S. Thrun, T. Mitchell (2000). Text classification
    from labeled and unlabeled documents using EM. Machine Learning 39(2-3),
    pp. 103-134.
    """

    def __init__(self, alpha=1.0, beta=1.0, fit_prior=True, class_prior=None,
                 tol=1e-3, max_iter=20, verbose=True):
        self.alpha = alpha
        self.beta = beta
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _count(self, X, Y, U_X=np.array([]), U_prob=np.array([])):
        """Count and smooth feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        self.feature_count_ = safe_sparse_dot(Y.T, X)
        self.class_count_ = Y.sum(axis=0)

        if U_X.shape[0] > 0:
            self.feature_count_ += self.beta*safe_sparse_dot(U_prob.T, U_X)
            self.class_count_ += self.beta*U_prob.sum(axis=0)
        else:
            self.feature_count_ = safe_sparse_dot(Y.T, X)
            self.class_count_ = Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')
        return (safe_sparse_dot(X, self.feature_log_prob_.T) +
                self.class_log_prior_)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """A semi-supervised version of this method has not been implemented.
        """

    def fit(self, X, y, sample_weight=None):
        """Fit semi-supervised Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.  Unlabeled data must be marked with -1.

        sample_weight : array-like, shape = [n_samples], (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape
        # Unlabeled data are marked with -1
        unlabeled = np.flatnonzero(y == -1)
        labeled = np.setdiff1d(np.arange(len(y)), unlabeled)

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y[labeled])
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        Y = Y.astype(np.float64, copy=False)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_effective_classes = Y.shape[1]

        alpha = self._check_alpha()
        self._count(X[labeled], Y)


        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        jll = self._joint_log_likelihood(X)
        sum_jll = jll.sum()

        # Run EM algorithm
        if len(unlabeled) > 0:
            self.num_iter = 0
            pred = self.predict(X)
            while self.num_iter < self.max_iter:
                self.num_iter += 1
                prev_sum_jll = sum_jll

                # First, the E-step:
                prob = self.predict_proba(X[unlabeled])

                # Then, the M-step:
                self._count(X[labeled], Y, X[unlabeled], prob)
                self._update_feature_log_prob(self.beta)
                self._update_class_log_prior(class_prior=class_prior)

                jll = self._joint_log_likelihood(X)
                sum_jll = jll.sum()
                if self.verbose:
                    print(
                        'Step {}: jll = {:f}'.format(self.num_iter, sum_jll)
                    )

                if self.num_iter > 1 and prev_sum_jll - sum_jll < self.tol:
                    break

            if self.verbose:
                print(
                    'Optimization converged after {} '
                    'iteration'.format(self.num_iter)
                    + 's.' if self.num_iter > 1 else '.'
                )

        return self
