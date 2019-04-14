#include "test_functions.h"

#include <cmath>

namespace func {

// Current values are only for testing, for common use choose the way to assign l_i-s you like
constexpr double L1 = 1.0;
constexpr double L2 = 1.0;
constexpr double L3 = 1.0;

double u(double3 x, double t) {
    return std::exp(3 * t + x[0] + x[1] + x[2]);
}

double u0(double3 x) {
    return u(x, 0);
}

double a0(double2 x, double t) {
    return u({0, x[0], x[1]}, t);
}

double a1(double2 x, double t) {
    return u({L1, x[0], x[1]}, t);
}

double b0(double2 x, double t) {
    return u({x[0], 0, x[1]}, t);
}

double b1(double2 x, double t) {
    return u({x[0], L2, x[1]}, t);
}

double c0(double2 x, double t) {
    return u({x[0], x[1], 0}, t);
}

double c1(double2 x, double t) {
    return u({x[0], x[1], L3}, t);
}

} // namespace func
