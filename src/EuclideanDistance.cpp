#include "EuclideanDistance.h"
#include <cmath>
#include <stdexcept>
using namespace std;

double EuclideanDistance::calculate(const vector<double>& a, const vector<double>& b) const {
    if (a.size() != b.size()) {
        throw invalid_argument("EuclideanDistance: vectors must have the same size");
    }

    double sum = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sqrt(sum);
}