#pragma once
#include "IDistance.h"

class MinkowskiDistance : public IDistance {
private:
    double p;
public:
    explicit MinkowskiDistance(double p = 2.0);
    double calculate(const vector<double>& a, const vector<double>& b) const override;
};