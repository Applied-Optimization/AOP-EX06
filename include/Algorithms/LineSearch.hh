#pragma once

#include <FunctionBase/FunctionBaseSparse.hh>
#include <iostream>

//== NAMESPACES ===============================================================

namespace AOPT {

    class LineSearch {
    public:
        typedef FunctionBaseSparse::Vec Vec;
        typedef FunctionBaseSparse::SMat SMat;

        /** Backtracking line search method
         *
         * \param _problem a pointer to a specific Problem, which can be any type that
         *        has the same interface as FunctionBase's (i.e., with eval_f, eval_gradient, etc.)
         * \param _x starting point of the method. Should be of the same dimension as the Problem's
         * \param _g gradient at the starting point.
         * \param _dx delta x
         * \param _t0 initial step of the method
         * \param _alpha and _tau variation constant, as stated by the method's definition
         * \return the final step t computed by the backtracking line search */
        template <class Problem>
        static double backtracking(Problem *_problem,
                                   const Vec &_x,
                                   const Vec &_dx,
                                   const Vec &_g,
                                   const double _t0 = 1.0,
                                   const double _alpha = 0.5,
                                   const double _tau = 0.75) {

            double t = _t0;

            // Pre-compute objective function value at the starting point
            double fx = _problem->eval_f(_x);

            // Compute the dot product of gradient and delta_x
            double gtdx = _g.dot(_dx);

            // Ensure that _dx points in a descent direction
            if (gtdx > 0) {
                std::cerr << "Warning: _dx points in the direction that increases function value. gTdx = " << gtdx << std::endl;
                return t;
            }

            // Backtracking line search loop
            int i = 0;
            while (_problem->eval_f(_x + t * _dx) > fx + _alpha * t * gtdx && i < 1000) {
                t *= _tau;
                i++;
            }

            return t;
        }
    };
    
//=============================================================================
}
