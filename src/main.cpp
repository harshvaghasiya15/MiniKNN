#include <iostream>
#include <memory>
#include <iomanip>
#include "DataSet.h"
#include "DataPoint.h"
#include "EuclideanDistance.h"
#include "ManhattanDistance.h"
#include "MinkowskiDistance.h"
#include "KNNClassifier.h"
#include "Evaluator.h"
#include "Normalizer.h"

using namespace std;

int main() {
    try {
        // ---- 1. Load the full dataset ----
        DataSet all;
        all.loadCSV("data/iris.csv");
        cout << "Loaded " << all.size() << " rows." << endl;

        // ---- 2. Split into train/test (80/20, fixed seed for reproducibility) ----
        auto [trainRaw, testRaw] = Evaluator::trainTestSplit(all, 0.8, 42);
        cout << "Train: " << trainRaw.size() << " rows, Test: " << testRaw.size() << " rows." << endl;

        // ---- 3. Normalize: fit only on training data, apply same scale to test data ----
        Normalizer norm;
        DataSet trainScaled = norm.fitTransform(trainRaw);
        DataSet testScaled = norm.transform(testRaw);

        cout << "\nBefore normalization, first training point:" << endl;
        trainRaw.points[0].print();
        cout << "After normalization:" << endl;
        trainScaled.points[0].print();

        // ---- 4. Train and evaluate with Euclidean distance ----
        cout << "\n--- Euclidean, k=5 ---" << endl;
        KNNClassifier clfEuclid(5, make_unique<EuclideanDistance>());
        clfEuclid.fit(trainScaled);

        double accEuclid = Evaluator::accuracy(clfEuclid, testScaled);
        cout << "Accuracy: " << accEuclid * 100 << "%" << endl;

        auto matrixEuclid = Evaluator::confusionMatrix(clfEuclid, testScaled);
        Evaluator::printConfusionMatrix(matrixEuclid);

        // ---- 5. Train and evaluate with Manhattan distance, same split, for comparison ----
        cout << "\n--- Manhattan, k=5 ---" << endl;
        KNNClassifier clfManhattan(5, make_unique<ManhattanDistance>());
        clfManhattan.fit(trainScaled);

        double accManhattan = Evaluator::accuracy(clfManhattan, testScaled);
        cout << "Accuracy: " << accManhattan * 100 << "%" << endl;

        auto matrixManhattan = Evaluator::confusionMatrix(clfManhattan, testScaled);
        Evaluator::printConfusionMatrix(matrixManhattan);

        // ---- 6. Minkowski distance across several p values ----
        // p=1 should match Manhattan, p=2 should match Euclidean -- a nice sanity check,
        // plus it shows how accuracy shifts for values in between and beyond.
        cout << "\n--- Minkowski, k=5, comparing p values ---" << endl;
        vector<double> pValues = {1.0, 1.5, 2.0, 3.0, 4.0};

        cout << left << setw(8) << "p" << "Accuracy" << endl;
        for (double p : pValues) {
            KNNClassifier clfMinkowski(5, make_unique<MinkowskiDistance>(p));
            clfMinkowski.fit(trainScaled);
            double acc = Evaluator::accuracy(clfMinkowski, testScaled);
            cout << left << setw(8) << p << fixed << setprecision(2) << acc * 100 << "%" << endl;
        }

        // ---- 7. Predict a single new point, same way a real user would ----
        cout << "\n--- Single prediction test ---" << endl;
        DataPoint newFlower({5.0, 3.4, 1.5, 0.2}, "");   // unlabeled point
        DataPoint newFlowerScaled = norm.transform(newFlower);
        string predicted = clfEuclid.predict(newFlowerScaled);
        cout << "Predicted species: " << predicted << endl;

    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }

    return 0;
}