# MiniKNN

**KNN Classifier Library - C++ OOP Project**

## Problem Summary

K-Nearest Neighbours (KNN) classifies a new data point by looking at the *k* closest labelled points in the training data and returning the label that most of those neighbours share (majority vote).

MiniKNN is a small C++ library that does this using object-oriented design. The distance formula is not hardcoded. It is a swappable component (Euclidean or Manhattan), so the classifier can change how "closeness" is measured without changing its own code.

**Dataset:** Iris dataset, 150 rows. Each row has 4 numeric features (sepal length, sepal width, petal length, petal width) and a label (`Iris-setosa`, `Iris-versicolor` or `Iris-virginica`).

## Planned Structure

### Classes

| Class | Role |
|---|---|
| `DataPoint` | One row of data: a `vector<double>` of features and a `string` label. |
| `DataSet` | A collection of `DataPoint`s. Loads the Iris CSV file with `loadCSV(path)`. |
| `IDistance` | Abstract interface for a distance metric: `calculate(a, b)`. |
| `EuclideanDistance` | Implements `IDistance` using the straight-line distance. |
| `ManhattanDistance` | Implements `IDistance` using the sum of absolute differences. |
| `KNNClassifier` | Holds `k`, a distance metric (`unique_ptr<IDistance>`) and the training data. `fit(data)` stores the training set and `predict(p)` returns the majority label of the *k* nearest neighbours. |

### Relationships

- `EuclideanDistance` and `ManhattanDistance` inherit from `IDistance`.
- `DataSet` owns many `DataPoint`s (composition).
- `KNNClassifier` owns its `IDistance` metric and its training `DataSet` (composition).
- `KNNClassifier::predict` uses a `DataPoint` as its input (dependency).

The full diagram is in [`docs/class_diagram.png`](docs/class_diagram.png).

![Class diagram](docs/class_diagram.png)

### File layout

```
MiniKNN/
├── data/
│   └── iris.csv
├── docs/
│   └── class_diagram.png
├── include/
│   ├── DataPoint.h
│   ├── DataSet.h
│   ├── IDistance.h
│   ├── EuclideanDistance.h
│   ├── ManhattanDistance.h
│   └── KNNClassifier.h
├── src/
│   ├── DataPoint.cpp
│   ├── DataSet.cpp
│   ├── EuclideanDistance.cpp
│   ├── ManhattanDistance.cpp
│   ├── KNNClassifier.cpp
│   └── main.cpp
└── README.md
```

### OOP concepts demonstrated

- **Abstraction:** `IDistance` defines what a metric does, not how.
- **Inheritance:** the two metrics extend `IDistance`.
- **Polymorphism:** `KNNClassifier` calls `calculate()` through an `IDistance` pointer, and the right formula runs at runtime.
- **Composition:** `KNNClassifier` owns its metric and training data.
- **Encapsulation:** `KNNClassifier` keeps `k`, the metric and the training data private.
- **RAII / smart pointers:** `unique_ptr` manages the metric's memory automatically.

## Build and Run

```
g++ -std=c++17 -Iinclude src/*.cpp -o miniknn
./miniknn
```

The program reads `data/iris.csv`, so run it from the project root folder.

## Team

| Member | Part |
|---|---|
| Harsh Vaghasiya | Data layer: `DataPoint`, `DataSet`, `iris.csv` |
| Deep Vekariya | Distance layer: `IDistance`, `EuclideanDistance`, `ManhattanDistance` |
| Sanyam kocher | Classifier interface (`KNNClassifier.h`), class diagram, README |


