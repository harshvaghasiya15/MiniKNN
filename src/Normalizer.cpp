#include"Normalizer.h"
#include<stdexcept>
#include<limits>
using namespace std;

void Normalizer::fit(const DataSet& data){
    if(data.size() == 0){
        throw invalid_argument("Normalizer: cannot fit on an empty dataset");
    }

    size_t featureCount = data.points[0].features.size();

    minVals.assign(featureCount, numeric_limits<double>::infinity());
    maxVals.assign(featureCount, -numeric_limits<double>::infinity());

    for(const DataPoint& p : data.points){
        if(p.features.size() != featureCount){
            throw invalid_argument("Normalizer: all points must have the same number of features");
        }
        for(size_t i = 0; i < featureCount; i++){
            if(p.features[i] < minVals[i]) minVals[i] = p.features[i];
            if(p.features[i] > maxVals[i]) maxVals[i] = p.features[i];
        }
    }

    fitted = true;
}

DataPoint Normalizer::transform(const DataPoint& p) const{
    if(!fitted){
        throw runtime_error("Normalizer: call fit() before transform()");
    }
    if(p.features.size() != minVals.size()){
        throw invalid_argument("Normalizer: point has a different number of features than fit() saw");
    }

    vector<double> scaled(p.features.size());
    for(size_t i = 0; i < p.features.size(); i++){
        double range = maxVals[i] - minVals[i];
        if(range == 0.0){
            scaled[i] = 0.0;
        }
        else{
            scaled[i] = (p.features[i] - minVals[i]) / range;
        }
    }

    return DataPoint(scaled, p.label);
}

DataSet Normalizer::transform(const DataSet& data) const{
    DataSet result;
    result.points.reserve(data.size());
    for(const DataPoint& p : data.points){
        result.points.push_back(transform(p));
    }
    return result;
}

DataSet Normalizer::fitTransform(const DataSet& data){
    fit(data);
    return transform(data);
}