#pragma once

#include <array>

using double3 = std::array<double, 3>;
using double2 = std::array<double, 2>;

namespace func {

/*
This function is used to measure the quality of approximation method
and to define conditions but in common way the implementation must differ from current.
*/
double u(double3 x, double t);

double u0(double3 x);

double a0(double2 x, double t);

double a1(double2 x, double t);

double b0(double2 x, double t);

double b1(double2 x, double t);

double c1(double2 x, double t);

double c2(double2 x, double t);

} // namespace func
