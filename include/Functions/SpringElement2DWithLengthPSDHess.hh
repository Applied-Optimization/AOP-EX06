#pragma once

#include "SpringElement2DWithLength.hh"

#include <Eigen/Eigenvalues>
//== NAMESPACES ===============================================================

namespace AOPT {

//== CLASS DEFINITION =========================================================

/**
*   Class that overrides the hessian of the non-convex energy of the spring element
 * by fixing the negative eigen values of the hessian matrix
*/

class SpringElement2DWithLengthPSDHess : public SpringElement2DWithLength {
public:

    SpringElement2DWithLengthPSDHess(): SpringElement2DWithLength() {}

    inline virtual void eval_hessian(const Vec &_x, const Vec &_coeffs, Mat &_H) override {
        //------------------------------------------------------//
        //TODO: compute the hessian matrix and project it to a positve definite matrix
        //Hint: 1. to compute the eigen decomposition, use
        //          Eigen::SelfAdjointEigenSolver<Mat> solver(A);
        //          Mat evecs = solver.eigenvectors();  //this matrix contains the eigenvectors in its columns
        //          Vec evals = solver.eigenvalues();
        //      2. to convert a vector d to a (dense) diagonal matrix D, use
        //          D = d.asDiagonal()

        Eigen::SelfAdjointEigenSolver<Mat> solver(_H);
    Vec evals = solver.eigenvalues();
    Mat evecs = solver.eigenvectors();

    // Project eigenvalues to ensure positive definiteness
    for (int i = 0; i < evals.size(); ++i) {
        if (evals[i] < m_eps) {
            evals[i] = m_eps;
        }
    }

    // Reconstruct the positive definite Hessian matrix
    Mat D = evals.asDiagonal();
    _H = evecs * D * evecs.transpose();
        

        //------------------------------------------------------//
    }

    static constexpr double m_eps = 1e-7;
};

//=============================================================================

}
