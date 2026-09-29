#pragma once
#include "IDistance.h"
using namespace std;

class EuclideanDistance : public IDistance {
public:
    double calculate(const vector<double>& a, const vector<double>& b) const override;
};