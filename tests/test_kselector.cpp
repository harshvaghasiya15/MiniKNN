#include <iostream>
#include "KSelector.h"
#include "KNNClassifier.h"
#include "EuclideanDistance.h"
#include "ManhattanDistance.h"

int main() {
    DataSet all;
    all.loadCSV("data/iris.csv");

    vector<int> candidates{1, 3, 5, 7, 9, 11, 13, 15, 21};

    auto euclid = [] { return unique_ptr<IDistance>(make_unique<EuclideanDistance>()); };
    auto manhat = [] { return unique_ptr<IDistance>(make_unique<ManhattanDistance>()); };

    cout << "Best k (Euclidean): " << KSelector::findBestK(all, candidates, euclid) << "\n";
    cout << "Best k (Manhattan): " << KSelector::findBestK(all, candidates, manhat) << "\n";
}