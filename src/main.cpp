#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


#include "DataSet.h"
#include "DataPoint.h"
#include "IDistance.h"
#include "EuclideanDistance.h"
#include "ManhattanDistance.h"
#include "MinkowskiDistance.h"
#include "KNNClassifier.h"
#include "Evaluator.h"
#include "Normalizer.h"
#include "KSelector.h"

using namespace std;

// ---------------------------------------------------------------------------
// Settings the user can change from the menu
// ---------------------------------------------------------------------------
struct Settings {
    int metric = 1;     // 1 = Euclidean, 2 = Manhattan, 3 = Minkowski
    double p = 2.0;     // only used when metric == 3
    int k = 5;
};

// Everything the menu needs: the prepared data and the current settings
struct Session {
    DataSet trainScaled;
    DataSet testScaled;
    Normalizer norm;
    Settings settings;
};

// Thrown when the input stream ends (Ctrl+D / Ctrl+Z or piped input runs out)
class InputClosed : public runtime_error {
public:
    InputClosed() : runtime_error("input closed") {}
};

// ---------------------------------------------------------------------------
// Input helpers: they keep asking until the user types something valid,
// so a typo never crashes the program.
// ---------------------------------------------------------------------------
string readLine(const string& prompt) {
    cout << prompt;
    string line;
    if (!getline(cin, line)) {
        throw InputClosed();
    }
    return line;
}

bool parseInt(const string& text, int& value) {
    try {
        size_t pos = 0;
        int v = stoi(text, &pos);
        while (pos < text.size() && isspace(static_cast<unsigned char>(text[pos]))) pos++;
        if (pos != text.size()) return false;
        value = v;
        return true;
    } catch (const exception&) {
        return false;
    }
}

bool parseDouble(const string& text, double& value) {
    try {
        size_t pos = 0;
        double v = stod(text, &pos);
        while (pos < text.size() && isspace(static_cast<unsigned char>(text[pos]))) pos++;
        if (pos != text.size() || !isfinite(v)) return false;
        value = v;
        return true;
    } catch (const exception&) {
        return false;
    }
}

int readInt(const string& prompt, int low, int high) {
    while (true) {
        int value;
        if (parseInt(readLine(prompt), value) && value >= low && value <= high) {
            return value;
        }
        cout << "Please enter a whole number from " << low << " to " << high << "." << endl;
    }
}

double readDouble(const string& prompt) {
    while (true) {
        double value;
        if (parseDouble(readLine(prompt), value)) {
            return value;
        }
        cout << "Please enter a number (for example 5.1)." << endl;
    }
}

// ---------------------------------------------------------------------------
// Building a classifier from the current settings
// ---------------------------------------------------------------------------
unique_ptr<IDistance> makeMetric(const Settings& s) {
    switch (s.metric) {
        case 1:  return make_unique<EuclideanDistance>();
        case 2:  return make_unique<ManhattanDistance>();
        case 3:  return make_unique<MinkowskiDistance>(s.p);
        default: throw invalid_argument("Unknown metric choice");
    }
}

string metricName(const Settings& s) {
    switch (s.metric) {
        case 1:  return "Euclidean";
        case 2:  return "Manhattan";
        case 3: {
            ostringstream out;
            out << "Minkowski (p=" << s.p << ")";
            return out.str();
        }
        default: return "Unknown";
    }
}

KNNClassifier buildClassifier(const Session& session) {
    KNNClassifier clf(session.settings.k, makeMetric(session.settings));
    clf.fit(session.trainScaled);
    return clf;
}

// ---------------------------------------------------------------------------
// Menu actions
// ---------------------------------------------------------------------------
void chooseMetric(Session& session) {
    cout << "\n  1. Euclidean\n  2. Manhattan\n  3. Minkowski (you choose p)\n";
    int choice = readInt("Choose a metric (1-3): ", 1, 3);

    if (choice == 3) {
        double p;
        while (true) {
            p = readDouble("Enter p (must be 1 or more): ");
            if (p >= 1.0) break;
            cout << "p must be at least 1." << endl;
        }
        session.settings.p = p;
    }
    session.settings.metric = choice;
    cout << "Metric set to " << metricName(session.settings) << "." << endl;
}

void chooseK(Session& session) {
    int maxK = static_cast<int>(session.trainScaled.size());
    int k = readInt("Enter k (1-" + to_string(maxK) + "): ", 1, maxK);
    session.settings.k = k;
    cout << "k set to " << k << "." << endl;
}

void showAccuracy(const Session& session) {
    KNNClassifier clf = buildClassifier(session);

    double acc = Evaluator::accuracy(clf, session.testScaled);
    cout << "\nMetric: " << metricName(session.settings)
         << ", k = " << session.settings.k << endl;
    cout << "Test rows: " << session.testScaled.size() << endl;
    cout << "Accuracy: " << fixed << setprecision(2) << acc * 100 << "%" << endl;

    cout << "\nConfusion matrix (rows = actual, columns = predicted):" << endl;
    Evaluator::printConfusionMatrix(Evaluator::confusionMatrix(clf, session.testScaled));
}

void classifyCustomPoint(const Session& session) {
    static const vector<string> names = {
        "Sepal length (cm)", "Sepal width (cm)", "Petal length (cm)", "Petal width (cm)"
    };

    size_t featureCount = session.trainScaled.points[0].features.size();
    cout << "\nEnter the measurements of the flower:" << endl;

    vector<double> features;
    for (size_t i = 0; i < featureCount; i++) {
        string name = (i < names.size()) ? names[i] : "Feature " + to_string(i + 1);
        features.push_back(readDouble("  " + name + ": "));
    }

    KNNClassifier clf = buildClassifier(session);

    DataPoint raw(features, "");                           // unlabeled point
    DataPoint scaled = session.norm.transform(raw);        // same scaling as the training data
    string predicted = clf.predict(scaled);

    cout << "Predicted species: " << predicted
         << "  (" << metricName(session.settings) << ", k = " << session.settings.k << ")" << endl;
}

void findBestK(Session& session) {
    vector<int> candidates;
    for (int k = 5; k <= 45; k += 2) candidates.push_back(k);

    cout << "\nTrying k = 5,7,9, ... 41,43,45 with 5-fold cross-validation on the training data..." << endl;

    const Settings current = session.settings;
    int best = KSelector::findBestK(session.trainScaled, candidates,
                                    [current]() { return makeMetric(current); });

    session.settings.k = best;
    cout << "Best k for " << metricName(current) << ": " << best
         << " (k has been set to " << best << ")." << endl;
}

void printMenu(const Settings& s) {
    cout << "\n==================== MiniKNN ====================" << endl;
    cout << "Current: " << metricName(s) << ", k = " << s.k << endl;
    cout << "  1. Choose distance metric" << endl;
    cout << "  2. Set k" << endl;
    cout << "  3. Show accuracy on the test set" << endl;
    cout << "  4. Classify a custom flower" << endl;
    cout << "  5. Find the best k automatically" << endl;
    cout << "  0. Exit" << endl;
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    string path = (argc > 1) ? argv[1] : "data/iris.csv";
    Session session;

    try {
        // ---- 1. Load the dataset ----
        DataSet all;
        all.loadCSV(path);
        cout << "Loaded " << all.size() << " rows from " << path << "." << endl;

        // ---- 2. Split 80/20 with a fixed seed so results are repeatable ----
        auto [trainRaw, testRaw] = Evaluator::trainTestSplit(all, 0.8, 42);
        cout << "Train: " << trainRaw.size() << " rows, Test: " << testRaw.size() << " rows." << endl;

        // ---- 3. Normalise: learn the scale from the training data only ----
        session.trainScaled = session.norm.fitTransform(trainRaw);
        session.testScaled = session.norm.transform(testRaw);
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }

    // ---- 4. Menu loop: a mistake in one action never ends the program ----
    while (true) {
        try {
            printMenu(session.settings);
            int choice = readInt("Your choice: ", 0, 5);

            if (choice == 0) break;
            else if (choice == 1) chooseMetric(session);
            else if (choice == 2) chooseK(session);
            else if (choice == 3) showAccuracy(session);
            else if (choice == 4) classifyCustomPoint(session);
            else if (choice == 5) findBestK(session);
        } catch (const InputClosed&) {
            cout << endl;
            break;
        } catch (const exception& e) {
            cerr << "ERROR: " << e.what() << endl;
        }
    }

    cout << "Goodbye." << endl;
    return 0;
}
