#include <iostream>
#include "KNNClassifier.h"
#include "EuclideanDistance.h"
#include "ManhattanDistance.h"

int main() {
    DataSet d;
    d.points.push_back(DataPoint({1, 1}, "A"));
    d.points.push_back(DataPoint({1, 2}, "A"));
    d.points.push_back(DataPoint({2, 1}, "A"));
    d.points.push_back(DataPoint({8, 8}, "B"));
    d.points.push_back(DataPoint({9, 8}, "B"));
    d.points.push_back(DataPoint({8, 9}, "B"));

    // Euclidean
    KNNClassifier e(3, make_unique<EuclideanDistance>());
    e.fit(d);
    cout << "Euclidean: " << e.predict(DataPoint({2, 2})) << " (expect A)\n";
    cout << "Euclidean: " << e.predict(DataPoint({9, 9})) << " (expect B)\n";

    // Manhattan: same classifier code, different strategy
    KNNClassifier m(3, make_unique<ManhattanDistance>());
    m.fit(d);
    cout << "Manhattan: " << m.predict(DataPoint({2, 2})) << " (expect A)\n";
    cout << "Manhattan: " << m.predict(DataPoint({9, 9})) << " (expect B)\n";

    // Error handling
    try {
        KNNClassifier bad(0, make_unique<EuclideanDistance>());
    } catch (const invalid_argument& ex) {
        cout << "k=0 rejected: " << ex.what() << "\n";
    }
    try {
        KNNClassifier unfitted(3, make_unique<EuclideanDistance>());
        unfitted.predict(DataPoint({1, 1}));
    } catch (const runtime_error& ex) {
        cout << "predict before fit rejected: " << ex.what() << "\n";
    }
    try {
        KNNClassifier big(10, make_unique<EuclideanDistance>());
        big.fit(d);
        big.predict(DataPoint({1, 1}));
    } catch (const runtime_error& ex) {
        cout << "k too large rejected: " << ex.what() << "\n";
    }
}