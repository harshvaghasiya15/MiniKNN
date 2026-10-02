#include <algorithm>
#include <iostream>
#include <random>
#include "KNNClassifier.h"
#include "EuclideanDistance.h"
#include "ManhattanDistance.h"

double accuracyFor(unique_ptr<IDistance> metric, int k,
                   const DataSet& train, const DataSet& test) {
    KNNClassifier knn(k, std::move(metric));
    knn.fit(train);
    int correct = 0;
    for (const DataPoint& p : test.points)
        if (knn.predict(p) == p.label) correct++;
    return 100.0 * correct / test.size();
}

int main() {
    DataSet all;
    all.loadCSV("data/iris.csv");
    cout << "Loaded " << all.size() << " rows\n";

    vector<DataPoint> pts = all.points;
    mt19937 rng(42);                       // fixed seed = repeatable result
    shuffle(pts.begin(), pts.end(), rng);

    size_t cut = pts.size() * 80 / 100;    // 120 train, 30 test
    DataSet train, test;
    train.points.assign(pts.begin(), pts.begin() + cut);
    test.points.assign(pts.begin() + cut, pts.end());

    for (int k : {1, 3, 5, 7, 11, 15}) {
        cout << "k=" << k
             << "  Euclidean: " << accuracyFor(make_unique<EuclideanDistance>(), k, train, test) << "%"
             << "  Manhattan: " << accuracyFor(make_unique<ManhattanDistance>(), k, train, test) << "%\n";
    }
}