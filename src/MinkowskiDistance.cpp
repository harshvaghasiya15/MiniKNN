#include "MinkowskiDistance.h"
#include <cmath>
#include <stdexcept>
using namespace std;

MinkowskiDistance::MinkowskiDistance(double p) : p(p) {
    if (p < 1.0) {
        throw invalid_argument("MinkowskiDistance: p must be >= 1");
    }
}

double MinkowskiDistance::calculate(const vector<double>& a, const vector<double>& b) const {
    if (a.size() != b.size()) {
        throw invalid_argument("MinkowskiDistance: vectors must have the same size");
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += pow(fabs(a[i] - b[i]), p);
    }
    return pow(sum, 1.0 / p);
}
