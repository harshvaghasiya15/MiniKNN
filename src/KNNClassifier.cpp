#include "KNNClassifier.h"
#include <algorithm>
#include <map>
#include <stdexcept>
using namespace std;

KNNClassifier::KNNClassifier(int k, unique_ptr<IDistance> metric) : k(k), metric(std::move(metric)) {
    if (k <= 0) {
        throw invalid_argument("KNNClassifier: k must be positive");
    }
    if (!this->metric) {
        throw invalid_argument("KNNClassifier: metric must not be null");
    }
}


void KNNClassifier::fit(const DataSet& data) {
    if (data.size() == 0) {
        throw invalid_argument("KNNClassifier: training data is empty");
    }
    trainingData = data;
}


string KNNClassifier::predict(const DataPoint& p) const {
    if (trainingData.size() == 0) {
        throw runtime_error("KNNClassifier: call fit() before predict()");
    }
    if (static_cast<size_t>(k) > trainingData.size()) {
        throw runtime_error("KNNClassifier: k is larger than the training set");
    }


    vector<pair<double, string>> dist;
    for (const DataPoint& t : trainingData.points) {
        dist.push_back({p.distanceTo(t, *metric), t.label});
    }


    partial_sort(dist.begin(), dist.begin() + k, dist.end());


    map<string, int> votes;
    for (int i = 0; i < k; i++) {
        votes[dist[i].second]++;
    }


    string best = dist[0].second;
    int bestCount = votes[best];
    for (const auto& v : votes) {
        if (v.second > bestCount) {
            best = v.first;
            bestCount = v.second;
        }
    }
    return best;
}
