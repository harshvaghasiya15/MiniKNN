#include "ManhattanDistance.h"
#include <cmath>
#include <stdexcept>

using namespace std;

double ManhattanDistance::calculate(const vector<double>& a,
                                    const vector<double>& b) const {
    if (a.size() != b.size()) {
        throw invalid_argument("ManhattanDistance: vectors must have the same size");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += fabs(a[i] - b[i]);
    }
    return sum;
}