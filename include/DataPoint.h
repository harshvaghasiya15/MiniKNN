#ifndef DATAPOINT_H
#define DATAPOINT_H

#include"IDistance.h"
#include<vector>
#include<string>
using namespace std;

class DataPoint{
public:
    vector<double> features;
    string label;

    DataPoint();
    DataPoint(const vector<double>& features,const string& label = "");

    double distanceTo(const DataPoint& other,const IDistance& metric) const;

    void print() const;
};

#endif