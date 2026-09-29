#include"DataSet.h"
#include<iostream>
using namespace std;

int main(){
    DataSet ds;
    try{
        ds.loadCSV("data/iris.csv");
        cout << "Loaded " << ds.size() << " rows." << endl;
        ds.points[0].print();
    } catch(const exception& e){
        cerr << "ERROR: " << e.what() << endl;
    }
    return 0;
}