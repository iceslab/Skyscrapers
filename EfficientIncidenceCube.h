// This code is adapted from Java to C++ by Bartosz Ciesla
// Original Java code released under GPL v3 can be found here:
// https://github.com/bluemontag/igs-lsgp
// Author of original: Ignacio Gallego Sagastume

#pragma once
#include "asserts.h"
#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <iostream>
#include <iterator>
#include <functional>
#include <sstream>
#include <stdexcept>

typedef std::vector<int> lowerArrayT;
typedef std::vector<std::vector<lowerArrayT>> matrixT;
typedef std::tuple<int, int, int> orderedTripleT;

class EfficientIncidenceCube
{
public:
    EfficientIncidenceCube(int n);
    ~EfficientIncidenceCube() = default;

    int shuffle();

    int getEmptySpace(lowerArrayT &arr);
    int coordOf(int x, int y, int z);
    int plusOneZCoordOf(int x, int y);
    int secondPlusOneZCoordOf(int x, int y);
    int plusOneCoordOf(const matrixT& matrix, int first, int second);
    int plusOneXCoordOf(int y, int z);
    int plusOneYCoordOf(int x, int z);
    int minusOneCoordOf(int x, int y);
    void doPlusMinus1Move(orderedTripleT t, int x1, int y1, int z1);
    void moveFromProper();
    void moveFromImproper();
    int choosePlusOneCoordOf(const matrixT& matrix, int first, int second);
    int choosePlusOneZCoordOf(int x, int y);
    int choosePlusOneXCoordOf(int y, int z);
    int choosePlusOneYCoordOf(int x, int z);

    std::string toString();
    std::string printRaw();

protected:
    void xyzStore(int x, int y, int z);
    void xyzRemove(int x, int y, int z);
    orderedTripleT select0Cell();

    const int n;
    matrixT xyMatrix;	//maximum of 3 elements in the row or column (-z, z, t)
    matrixT yzMatrix;
    matrixT xzMatrix;
    bool proper;
    orderedTripleT improperCell;
private:
    int add(lowerArrayT& arr, int elem);
    int remove(lowerArrayT& arr, int elem);
    int minus(int a);

    static const int nullInt = -99999; //a number outside the scope of symbols
    static const int minus0 = -10000;
    //Each view is stored as a two-dimensional array, 
    //to avoid sequential searches for "1" elements along the cube
    //the lists store all possible values of the third coordinate
    static const int max = 3;

    static std::random_device r;
    static std::default_random_engine r_engine;
};

