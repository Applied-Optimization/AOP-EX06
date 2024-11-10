#pragma once

#include <FunctionBase/FunctionBaseSparse.hh>
#include "LineSearch.hh"

//== NAMESPACES ===============================================================

namespace AOPT {
    /**
    * @brief NewtonMethods is just a list of functions implementing several variations of the
    * newton's method */
    class NewtonMethods {
    public:
        typedef FunctionBaseSparse::Vec Vec;   // dense vector arbitrary size
        typedef FunctionBaseSparse::Mat Mat;   // dense matrix arbitrary size
        typedef FunctionBaseSparse::T T;        //Triplets
        typedef FunctionBaseSparse::SMat SMat;  // sparse matrix arbitrary size

        /**
         * @brief solve
         * \param _problem pointer to any function/problem inheriting from FunctionBaseSparse
         *        on which the basic Newton Method will be applied
         * \param _initial_x starting point of the method
         * \param _eps epsilon under which the method stops
         * \param _max_iters maximum iteration of the method*/
        static Vec solve(FunctionBaseSparse *_problem, const Vec& _initial_x, const double _eps = 1e-4, const int _max_iters = 1000000) {
            std::cout << "******** Newton Method ********" << std::endl;

            // squared epsilon for stopping criterion
            double e2 = 2* _eps * _eps;

            int n = _problem->n_unknowns();

            // get starting point
            Vec x = _initial_x;

            // allocate gradient storage
            Vec g(n);

            // allocate hessian storage
            SMat H(n, n);

            // allocate search direction vector storage
            Vec delta_x(n);
            int iter = 0;


            Eigen::SimplicialLLT<SMat> solver;
  
            //------------------------------------------------------//
            //TODO: implement Newton method
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


                double t0 = 1.0;      // Initial step size
                double alpha = 0.5;   // Decrease condition factor
                double tau = 0.75;    // Scaling factor

                double step_size = LineSearch::backtracking_line_search(_problem, x, g, delta_x, t0, alpha, tau);

                //double step_size = LineSearch::backtracking_line_search(_problem, x, delta_x, g);

                x += step_size * delta_x;



                ++iter;

            }


            if (iter == _max_iters) {

                std::cout << "Reached maximum iterations without convergence." << std::endl;

            }
           
            //------------------------------------------------------//

            return x;
        }

        /**
         * @brief solve with the Projected Hessian method
         * \param _problem pointer to any function/problem inheriting from FunctionBaseSparse.
         *        This problem MUST provide a working eval_hession() function for this method to work.
         *
         * \param _initial_x starting point of the method
         * \param _tau_factor the evolution factor of the tau coefficient
         * \param _eps epsilon under which the method stops
         * \param _max_iters maximum iteration of the method*/
        static Vec solve_with_projected_hessian(FunctionBaseSparse *_problem, const Vec& _initial_x, const double _gamma = 10.0,
                                                const double _eps = 1e-4, const int _max_iters = 1000000) {
            bool converged = false;
            return solve_with_projected_hessian(_problem, converged, _initial_x, _gamma, _eps, _max_iters);
        }

        static Vec solve_with_projected_hessian(FunctionBaseSparse *_problem, bool& _converged, const Vec& _initial_x, const double _gamma = 10.0,
                                                const double _eps = 1e-4, const int _max_iters = 1000000) {
            std::cout << "******** Newton Method with projected hessian ********" << std::endl;

            // squared epsilon for stopping criterion
            double e2 = 2*_eps * _eps;

            int n = _problem->n_unknowns();

            // get starting point
            Vec x = _initial_x;

            // allocate gradient storage
            Vec g(n);

            // allocate hessian storage
            SMat H(n, n);

            // allocate search direction vector storage
            Vec delta_x(n);
            int iter = 0;

            // identity and scalar to add positive values to the diagonal
            SMat I(n, n);
            I.setIdentity();

            _converged = false;

            Eigen::SimplicialLLT<SMat> solver;

            //------------------------------------------------------//
            //TODO: implement Newton with projected hessian method
            //Hint: if the factorization fails, then add delta * I to the hessian.
            //      repeat until factorization succeeds (make sure to update delta!)
            while (iter < _max_iters) {

                _problem->eval_gradient(x, g);

                _problem->eval_hessian(x, H);



                if (g.squaredNorm() <= e2) {

                    std::cout << "Converged after " << iter << " iterations." << std::endl;

                    _converged = true;

                    break;

                }



                bool factorization_success = false;
                double delta = 1e-4;  // Scalar for regularization

                while (!factorization_success) {

                    solver.compute(H + delta * I);

                    if (solver.info() == Eigen::Success) {

                        factorization_success = true;

                    } else {

                        delta *= _gamma;

                    }

                }



                delta_x = solver.solve(-g);

                double t0 = 1.0;      // Initial step size
                double alpha = 0.5;   // Decrease condition factor
                double tau = 0.75;    // Scaling factor

                double step_size = LineSearch::backtracking_line_search(_problem, x, g, delta_x, t0, alpha, tau);

                //double step_size = LineSearch::backtracking_line_search(_problem, x, delta_x, g);

                x += step_size * delta_x;



                ++iter;

            }



            if (iter == _max_iters) {

                std::cout << "Reached maximum iterations without convergence." << std::endl;

            }
           
            //------------------------------------------------------//


            return x;
        }
        



    };

} // namespace AOPT
