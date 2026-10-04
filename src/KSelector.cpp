#include "KSelector.h"
#include "KNNClassifier.h"
#include <algorithm>
#include <random>
#include <stdexcept>
using namespace std;

int KSelector::findBestK(const DataSet& train, const vector<int>& candidates,  function<unique_ptr<IDistance>()> makeMetric,
                         int folds, unsigned seed) {
    if (candidates.empty() || folds < 2 || train.size() < static_cast<size_t>(folds)) {
        throw invalid_argument("KSelector: bad arguments");
    }

    vector<DataPoint> shuffled = train.points;
    mt19937 rng(seed);
    shuffle(shuffled.begin(), shuffled.end(), rng);

    int bestK = -1;
    double bestAcc = -1.0;

    for (int k : candidates) {
        int correct = 0, total = 0;
        bool valid = true;

        for (int f = 0; f < folds; f++) {
            DataSet fitSet;
            vector<DataPoint> validation;
            for (size_t i = 0; i < shuffled.size(); i++) {
                if (static_cast<int>(i % folds) == f) validation.push_back(shuffled[i]);
                else fitSet.points.push_back(shuffled[i]);
            }
            if (k <= 0 || static_cast<size_t>(k) > fitSet.size()) { valid = false; break; }

            KNNClassifier knn(k, makeMetric());
            knn.fit(fitSet);
            for (const DataPoint& v : validation) {
                if (knn.predict(v) == v.label) correct++;
                total++;
            }
        }

        if (!valid || total == 0) continue;
        double acc = static_cast<double>(correct) / total;
        if (acc > bestAcc) {          
            bestAcc = acc;
            bestK = k;
        }
    }

    if (bestK == -1) throw runtime_error("KSelector: no valid k in candidates");
    return bestK;
}