#ifndef DATAPOINT_H
#define DATAPOINT_H

#include<vector>
#include<string>
using namespace std;

class DataPoint{
public:
    vector<double> features;
    string label;

    DataPoint();
    DataPoint(const vector<double>& features,const string& label = "");

    void print() const;
};

#endif