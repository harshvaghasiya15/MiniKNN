#ifndef NORMALIZER_H
#define NORMALIZER_H

#include<vector>
#include"DataSet.h"
#include"DataPoint.h"
using namespace std;

class Normalizer{
private:
    vector<double> minVals;
    vector<double> maxVals;
    bool fitted = false;

public:
    void fit(const DataSet& data);

    DataSet transform(const DataSet& data) const;

    DataPoint transform(const DataPoint& p) const;

    DataSet fitTransform(const DataSet& data);
};

#endif