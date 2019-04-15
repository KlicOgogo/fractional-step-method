#include "test_functions.h"

#include <cmath>

namespace func {

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
    return u({consts::l, x[0], x[1]}, t);
}

double b0(double2 x, double t) {
    return u({x[0], 0, x[1]}, t);
}

double b1(double2 x, double t) {
    return u({x[0], consts::l, x[1]}, t);
}

double c0(double2 x, double t) {
    return u({x[0], x[1], 0}, t);
}

double c1(double2 x, double t) {
    return u({x[0], x[1], consts::l}, t);
}

} // namespace func
