#include"DataPoint.h"
#include<iostream>
#include<cmath>
#include<stdexcept>
using namespace std;

DataPoint::DataPoint() {}

DataPoint::DataPoint(const vector<double>& features,const string& label) : features(features) , label(label) {}

void DataPoint::print() const{
    cout << "[";
    for(size_t i = 0;i < features.size();i++){
        cout << features[i];
        if(i != features.size() - 1) cout << ", ";
    }
    cout << "] -> " << (label.empty() ? "(Unlabeled)" : label) << endl;
}
