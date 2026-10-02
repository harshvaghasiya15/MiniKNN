#include"DataSet.h"
#include<fstream>
#include<sstream>
#include<iostream>
#include<stdexcept>
using namespace std;

void DataSet::loadCSV(const string& path){
    ifstream file(path);

    if(!file.is_open()){
        throw runtime_error("DataSet: Could not open file: " + path);
    }

    points.clear();
    string line;
    int lineNumber = 0;

    while(getline(file,line)){
        lineNumber++;
        
        if(line.empty()){
            continue;
        }

        stringstream ss(line);
        string cell;
        vector<double> features;
        string label;

        vector<string> cells;
        while(getline(ss,cell,',')){
            cells.push_back(cell);
        }

        if(cells.size() != 5){
            throw runtime_error("DataSet: Malformed row at line " + to_string(lineNumber) + ": expected 5 colums, got " + to_string(cells.size()));
        }

        try{
            for(int i = 0;i < 4;i++){
                features.push_back(stod(cells[i]));
            }
        } catch(const invalid_argument&){
            throw runtime_error("DataSet: Non numeric value at line: " + to_string(lineNumber));
        } catch(const out_of_range&){
            throw runtime_error("DataSet: Numeric value out of range at line: " + to_string(lineNumber));
        }

        label = cells[4];

        points.push_back(DataPoint(features,label));
    }

    file.close();

    if(points.empty()){
        throw runtime_error("DataSet: No data loaded from file: " + path);
    }
}

size_t DataSet::size() const{
    return points.size();
}

void DataSet::print() const{
    for(const auto& p:points){
        p.print();
    }
}