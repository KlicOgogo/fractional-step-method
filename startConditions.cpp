#include "startConditions.h"
#include <math.h>

double a0(double x2, double x3, double t) {
    if (t == 0) {
        return u0(0, x2, x3);
    }
    return exp(3*t + x2 + x3);
}

double a1(double maxX1, double x2, double x3, double t) {
    if (t == 0) {
        return u0(maxX1, x2, x3);
    }
    return exp(3*t + maxX1 + x2 + x3);
}

double b0(double x1, double x3, double t) {
    if (t == 0) {
        return u0(x1, 0, x3);
    }
    return exp(3*t + x1 + x3);
}

double b1(double maxX2, double x1, double x3, double t) {
    if (t == 0) {
        return u0(x1, maxX2, x3);
    }
    return exp(3*t + maxX2 + x1 + x3);
}

double c0(double x1, double x2, double t) {
    if (t == 0) {
        return u0(x1, x2, 0);
    }
    return exp(3*t + x1 + x2);
}

double c1(double maxX3, double x1, double x2, double t) {
    if (t == 0) {
        return u0(x1, x2, maxX3);
    }
    return exp(3*t + maxX3 + x1 + x2);
}

double u0(double x1, double x2, double x3) {
    return exp(x1+x2+x3);
}
