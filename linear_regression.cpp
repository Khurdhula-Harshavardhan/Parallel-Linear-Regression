#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <omp.h>

namespace py = pybind11;

Eigen::VectorXd fit_linear_regression(const Eigen::MatrixXd &X, const Eigen::VectorXd &y) {
    Eigen::MatrixXd X_b(X.rows(), X.cols() + 1);
    X_b << Eigen::VectorXd::Ones(X.rows()), X;

    Eigen::MatrixXd XtX = X_b.transpose() * X_b;
    Eigen::VectorXd Xty = X_b.transpose() * y;
    Eigen::VectorXd theta = XtX.ldlt().solve(Xty);

    return theta;
}

PYBIND11_MODULE(linear_regression, m) {
    m.def("fit_linear_regression", &fit_linear_regression, "A function that fits a linear regression model using the normal equation and OpenMP.");
}
