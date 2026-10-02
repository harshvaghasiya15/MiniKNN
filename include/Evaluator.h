#ifndef EVALUATOR.H
#define EVALUATOR.H

#include<utility>
#include<map>
#include<string>
#include"DataSet.h"
#include"KNNClassifier.h"
using namespace std;

class Evaluator{
public:
    static pair<DataSet,DataSet> trainTestSplit(const DataSet& data,double trainRatio,unsigned seed);

    static double accuracy(const KNNClassifier& classifier,const DataSet& testData);

    static map<string,map<string,int>> confusionMatrix(const KNNClassifier& classifier,const DataSet& testData);

    static void printConfusionMatrix(const map<string,map<string,int>>& matrix);
};

#endif