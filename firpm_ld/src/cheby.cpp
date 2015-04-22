/*
 * cheby.cpp
 *
 *  Created on: Sep 26, 2013
 *      Author: silviu
 */

#include "firpm/cheby.h"

void applyCos(std::vector<long double>& out,
        std::vector<long double> const& in)
{
    for (std::size_t i{0u}; i < in.size(); ++i)
        out[i] = cosl(in[i]);
}

void changeOfVariable(std::vector<long double>& out,
        std::vector<long double> const& in,
        long double& a, long double& b)
{
    for (std::size_t i{0u}; i < in.size(); ++i)
        out[i] = (b + a) / 2 + in[i] * (b - a) / 2;
}

void evaluateClenshaw(long double &result, std::vector<long double> &p,
        long double &x, long double &a, long double &b)
{
    long double bn1, bn2, bn;
    long double buffer;

    bn1 = 0;
    bn2 = 0;

    // compute the value of (2*x - b - a)/(b - a) in the temporary
    // variable buffer
    buffer = (x * 2 - b - a) / (b - a);

    // compute values of the coefficients
    int n = (int)p.size() - 1;
    for(int k{n}; k >= 0; --k) {
        bn = buffer * 2;
        bn = bn * bn1 - bn2 + p[k];
        // update values
        bn2 = bn1;
        bn1 = bn;
    }

    // set the value for the result (i.e. the value of the CI at x)
    result = bn1 - buffer * bn2;
}

void evaluateClenshaw(long double &result, std::vector<long double> &p,
                            long double &x)
{
    long double bn1, bn2, bn;

    int n = (int)p.size() - 1;
    bn2 = 0;
    bn1 = p[n];
    for(int k{n - 1}; k >= 1; --k) {
        bn = x * 2;
        bn = bn * bn1 - bn2 + p[k];
        // update values
        bn2 = bn1;
        bn1 = bn;
    }

    result = x * bn1 - bn2 + p[0];
}

void evaluateClenshaw2ndKind(long double &result, std::vector<long double> &p,
                            long double &x)
{
    long double bn1, bn2, bn;

    int n = (int)p.size() - 1;
    bn2 = 0;
    bn1 = p[n];
    for(int k{n - 1}; k >= 1; --k) {
        bn = x * 2;
        bn = bn * bn1 - bn2 + p[k];
        // update values
        bn2 = bn1;
        bn1 = bn;
    }

    result = (x * 2) * bn1 - bn2 + p[0];
}


void generateEquidistantNodes(std::vector<long double>& v, std::size_t n)
{
    // store the points in the vector v as v[i] = i * pi / n
    for(std::size_t i{0u}; i <= n; ++i) {
        v[i] = M_PI * i;
        v[i] /= n;
    }
}

void generateChebyshevPoints(std::vector<long double>& x, std::size_t n)
{
    // n is the number of points - 1
    x.reserve(n + 1u);
    if(n > 0u)
    {
        for(int k = n; k >= -n; k -= 2)
            x.push_back(sinl(M_PI * k / (n * 2)));
    }
    else
    {
        x.push_back(0);
    }
}

void generateChebyshevCoefficients(std::vector<long double>& c,
        std::vector<long double>& fv, std::size_t n)
{
    std::vector<long double> v(n + 1);
    generateEquidistantNodes(v, n);

    long double buffer;

    // halve the first and last coefficients
    long double oldValue1 = fv[0];
    long double oldValue2 = fv[n];
    fv[0] /= 2;
    fv[n] /= 2;

    for(std::size_t i{0u}; i <= n; ++i) {
        buffer = cos(v[i]);         // compute the actual value at the Chebyshev
                                    // node cos(i * pi / n)

        evaluateClenshaw(c[i], fv, buffer);

        if(i == 0u || i == n) {
            c[i] /= n;
        } else {
            c[i] *= 2;
            c[i] /= n;
        }
    }
    fv[0] = oldValue1;
    fv[n] = oldValue2;

}

// function that generates the coefficients of the derivative of a given CI
void derivativeCoefficients1stKind(std::vector<long double>& derivC,
                                        std::vector<long double>& c)
{
    int n = c.size() - 2;
    derivC[n] = c[n + 1] * (2 * (n + 1));
    derivC[n - 1] = c[n] * (2 * n);
    for(int i{n - 2}; i >= 0; --i) {
        derivC[i] = 2 * (i + 1);
        derivC[i] = derivC[i] * c[i + 1] + derivC[i + 2];
    }
    derivC[0] /= 2;
}

// use the formula (T_n(x))' = n * U_{n-1}(x)
void derivativeCoefficients2ndKind(std::vector<long double>& derivC,
        std::vector<long double>& c)
{
    std::size_t n = c.size() - 1;
    for(std::size_t i{n}; i > 0u; --i)
        derivC[i - 1] = c[i] * i;
}

