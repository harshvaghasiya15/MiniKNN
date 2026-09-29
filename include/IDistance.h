
#pragma once
#include <vector>
using  namespace std;

class IDistance {
public:
    virtual ~IDistance() = default;
    virtual double calculate(const vector<double>& a, const vector<double>& b) const = 0;
};