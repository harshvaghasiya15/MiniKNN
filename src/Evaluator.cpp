#include"Evaluator.h"
#include<algorithm>
#include<random>
#include<stdexcept>
#include<iostream>
#include<set>
#include <iomanip>
using namespace std;

pair<DataSet,DataSet> Evaluator::trainTestSplit(const DataSet& data, double trainRatio, unsigned seed){
    if(data.size() == 0){
        throw invalid_argument("Evaluator: cannot split an empty dataset");
    }

    if(trainRatio <=0.0 || trainRatio >= 1.0){
        throw invalid_argument("Evaluator: trainRatio must be strictly between 0 and 1");
    }

    vector<DataPoint> shuffled = data.points;

    mt19937 rng(seed);
    shuffle(shuffled.begin(),shuffled.end(),rng);

    size_t trainCount = static_cast<size_t>(shuffled.size() * trainRatio);

    if(trainCount == 0) trainCount = 1;
    if(trainCount == shuffled.size()) trainCount = shuffled.size() - 1;

    DataSet trainSet;
    DataSet testSet;
    trainSet.points.assign(shuffled.begin(), shuffled.begin() + trainCount);
    testSet.points.assign(shuffled.begin() + trainCount, shuffled.end());

    return {trainSet,testSet};
}

double Evaluator::accuracy(const KNNClassifier& classifier, const DataSet& testData){
    if(testData.size() == 0){
        throw invalid_argument("Evaluator: test data is empty");
    }

    int correct = 0;

    for(const DataPoint& p:testData.points){
        string predicted = classifier.predict(p);
        if(predicted == p.label){
            correct++;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(testData.size());
}

map<string, map<string, int>> Evaluator::confusionMatrix(const KNNClassifier& classifier, const DataSet& testData) {
    map<string, map<string, int>> matrix;

    for(const DataPoint& p : testData.points){
        string predicted = classifier.predict(p);
        matrix[p.label][predicted]++;
    }

    return matrix;
}

void Evaluator::printConfusionMatrix(const map<string, map<string, int>>& matrix){
    set<string> labels;
    for(const auto& row : matrix){
        labels.insert(row.first);
        for(const auto& col : row.second){
            labels.insert(col.first);
        }
    }

    const int colWidth = 20;

    cout << left << setw(colWidth) << "Actual \\ Predicted";
    for(const auto& l : labels) cout << setw(colWidth) << l;
    cout << endl;

    for(const auto& actual : labels){
        cout << left << setw(colWidth) << actual;
        for(const auto& predicted : labels){
            int count = 0;
            auto rowIt = matrix.find(actual);
            if(rowIt != matrix.end()){
                auto colIt = rowIt->second.find(predicted);
                if(colIt != rowIt->second.end()){
                    count = colIt->second;
                }
            }
            cout << setw(colWidth) << count;
        }
        cout << endl;
    }
}