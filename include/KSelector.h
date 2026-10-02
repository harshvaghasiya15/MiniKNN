#pragma once
#include <functional>
#include <memory>
#include <vector>
#include "DataSet.h"
#include "IDistance.h"

class KSelector {
public:

    static int findBestK(const DataSet& train, const vector<int>& candidates, function<unique_ptr<IDistance>()> makeMetric,
                        int folds = 5, unsigned seed = 42);
};