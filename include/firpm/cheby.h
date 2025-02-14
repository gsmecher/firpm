/**
 * @file cheby.h
 * @author Silviu Filip
 * @date 12 March 2015
 * @brief Useful functions for working with Chebyshev interpolants
 *
 */

//    firpm_d
//    Copyright (C) 2015 - 2019  S. Filip
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.

//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.

//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>

#ifndef CHEBY_H_
#define CHEBY_H_

#include "util.h"

/**
 * Gives the type of Chebyshev polynomial expansion to compute
 */
enum chebkind_t {
        FIRST,          /**< Chebyshev expansion of the first kind*/
        SECOND          /**< Chebyshev expansion of the second kind*/
};

/*! Computes the cosines of the elements of a vector
 * @param[in] in the vector to process
 * @param[out] out the vector containing the cosines of the elements from the
 * vector in
 */
template<typename T>
void cos(std::vector<T>& out,
        std::vector<T> const& in);

/*! Does a change of variable from the interval \f$\left[-1, 1\right]\f$ to the
 * interval \f$\left[a, b\right]\f$ on the elements of a given input vector
 * @param[in] in the vector of elements in \f$\left[-1, 1\right]\f$
 * @param[out] out the vector of elements after the change of variable
 * @param[in] a left bound of the desired interval
 * @param[in] b right bound of the desired interval
 */
template<typename T>
void chgvar(std::vector<T>& out,
        std::vector<T> const& in,
        T& a, T& b);

/*! Function that generates equidistant nodes inside the
 * \f$\left[0,\pi\right]\f$ interval, meaning values of the
 * form \f$\frac{i\pi}{n-1}\f$, where \f$0\leq i\leq n-1\f$
 * @param[out] v the vector that will contain the equi-distributed points
 * @param[in] n the number of points which will be computed
 */
template<typename T>
void equipts(std::vector<T>& v, std::size_t n);


/*! This function computes the values of the coefficients of the CI when
 * Chebyshev nodes of the second kind are used
 * @param[out] c vector used to hold the values of the computed coefficients
 * @param[in] fv vector that holds the value of the current function to
 * approximate at the Chebyshev nodes of the second kind scaled to the
 * appropriate interval (in our case it will be \f$\left[0,\pi\right]\f$)
 */
template<typename T>
void chebcoeffs(std::vector<T>& c,
                std::vector<T>& fv);

/*! Function that generates the coefficients of the derivative of a given CI
 *  @param[out] dc the vector of coefficients of the derivative of the CI
 *  @param[in] c the vector of coefficients of the CI whose derivative we
 *  want to compute
 *  @param[in] kind what kind of coefficient do we want to compute (for a
 *  Chebyshev expansion of the first or second kind)
 */
template<typename T>
void diffcoeffs(std::vector<T>& dc,
                std::vector<T>& c,
                chebkind_t kind = SECOND);

/*! Chebyshev proxy rootfinding method for a given CI
* @param[out] r the vector of computed roots of the CI
* @param[in] c the Chebyshev coeffients of the polynomial whose roots
* we want to find
* @param[in] dom the real domain where we are looking for the roots
* @param[in] kind the type of Chebyshev expansion (first or second)
* @param[in] balance flag signaling if we should use balancing (in 
* the vein of [Parlett&Reinsch1969] "Balancing a Matrix for
* Calculation of Eigenvalues and Eigenvectors") for the resulting 
* Chebyshev companion matrix 
*/
template<typename T>
void roots(std::vector<T>& r, std::vector<T>& c,
           std::pair<T, T> const &dom,
           chebkind_t kind = SECOND,
           bool balance = true);

#endif /* CHEBY_H_ */
