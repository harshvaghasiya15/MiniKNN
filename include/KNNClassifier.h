#ifndef KNNCLASSIFIER_H
#define KNNCLASSIFIER_H

#include<memory>
#include<string>
#include"DataPoint.h"
#include"DataSet.h"
#include"IDistance.h"
using namespace std;

class KNNClassifier{
private:
    int k;                          
    unique_ptr<IDistance> metric;   
    DataSet trainingData;           

public:
    KNNClassifier(int k, unique_ptr<IDistance> metric);

    void fit(const DataSet& data);
    string predict(const DataPoint& p) const;
};

#endif

