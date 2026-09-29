#ifndef DATASET_H
#define DATASET_H

#include<vector>
#include<string>
#include"DataPoint.h"
using namespace std;

class DataSet{
public:
    vector<DataPoint> points;

    void loadCSV(const string& path);

    size_t size() const;
    void print() const;

};

#endif