#pragma once

#include <FunctionBase/FunctionBaseSparse.hh>
#include "LineSearch.hh"
#include <iostream>
#include <Eigen/SparseCholesky>

//== NAMESPACES ===============================================================

namespace AOPT {

    class NewtonMethods {
    public:
        typedef FunctionBaseSparse::Vec Vec;   // dense vector of arbitrary size
        typedef FunctionBaseSparse::SMat SMat; // sparse matrix of arbitrary size

        static Vec solve(FunctionBaseSparse *_problem, const Vec& _initial_x, const double _eps = 1e-4, const int _max_iters = 1000000) {
            std::cout << "******** Newton Method ********" << std::endl;

            double e2 = 2 * _eps * _eps;
            int n = _problem->n_unknowns();
            Vec x = _initial_x;
            Vec g(n);
            SMat H(n, n);
            Vec delta_x(n);
            int iter = 0;
            Eigen::SimplicialLLT<SMat> solver;

            while (iter < _max_iters) {
                _problem->eval_gradient(x, g);
                _problem->eval_hessian(x, H);

                if (g.squaredNorm() <= e2) {
                    std::cout << "Converged after " << iter << " iterations." << std::endl;
                    break;
                }

                solver.compute(H);
                if (solver.info() != Eigen::Success) {
                    throw std::runtime_error("Hessian is not positive definite or cannot be decomposed.");
                }
                delta_x = solver.solve(-g);

                double step_size = LineSearch::backtracking(_problem, x, delta_x, g);
                x += step_size * delta_x;

                ++iter;
            }

            if (iter == _max_iters) {
                std::cout << "Reached maximum iterations without convergence." << std::endl;
            }

            return x;
        }

        static Vec solve_with_projected_hessian(FunctionBaseSparse *_problem, bool& _converged, const Vec& _initial_x, const double _gamma = 10.0,
                                                const double _eps = 1e-4, const int _max_iters = 1000000) {
            std::cout << "******** Newton Method with modified Hessian ********" << std::endl;

            double e2 = 2 * _eps * _eps;
            int n = _problem->n_unknowns();
            Vec x = _initial_x;
            Vec g(n);
            SMat H(n, n);
            Vec delta_x(n);
            int iter = 0;
            _converged = false;
            SMat I(n, n);
            I.setIdentity();

            _problem->eval_hessian(x, H);
            double delta = 1e-3 * H.diagonal().sum() / n;

            Eigen::SimplicialLLT<SMat> solver;

            while (iter < _max_iters) {
                _problem->eval_gradient(x, g);
                _problem->eval_hessian(x, H);

                if (g.squaredNorm() <= e2) {
                    std::cout << "Converged after " << iter << " iterations." << std::endl;
                    _converged = true;
                    break;
                }

                bool factorization_success = false;
                while (!factorization_success) {
                    solver.compute(H + delta * I);
                    if (solver.info() == Eigen::Success) {
                        factorization_success = true;
                    } else {
                        delta *= _gamma;
                    }
                }

                delta_x = solver.solve(-g);
                double step_size = LineSearch::backtracking(_problem, x, delta_x, g);
                x += step_size * delta_x;

                ++iter;
            }

            if (iter == _max_iters) {
                std::cout << "Reached maximum iterations without convergence." << std::endl;
            }

            return x;
        }

    };

} // namespace AOPT
