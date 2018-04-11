// This code is adapted from Java to C++ by Bartosz Ciesla
// Original Java code released under GPL v3 can be found here:
// https://github.com/bluemontag/igs-lsgp
// Author of original: Ignacio Gallego Sagastume

#include "EfficientIncidenceCube.h"

std::random_device EfficientIncidenceCube::r;
std::default_random_engine EfficientIncidenceCube::r_engine(r());

EfficientIncidenceCube::EfficientIncidenceCube(int n) : n(n), proper(true)
{
    xyMatrix.resize(n);
    yzMatrix.resize(n);
    xzMatrix.resize(n);

    for (int i = 0; i < n; i++)
    {
        xyMatrix[i].resize(n);
        yzMatrix[i].resize(n);
        xzMatrix[i].resize(n);
        for (int j = 0; j < n; j++)
        {
            xyMatrix[i][j].resize(max);
            yzMatrix[i][j].resize(max);
            xzMatrix[i][j].resize(max);
            for (int k = 0; k < max; k++)
            {
                xyMatrix[i][j][k] = nullInt;
                yzMatrix[i][j][k] = nullInt;
                xzMatrix[i][j][k] = nullInt;
            }
        }
    }

    int lastSymbol = -1;
    for (int i = 0; i < n; i++)
    {
        lastSymbol = (lastSymbol + 1) % n;
        for (int j = 0; j < n; j++)
        {
            xyzStore(i, j, lastSymbol);
            lastSymbol = (lastSymbol + 1) % n;
        }
    }
}

int EfficientIncidenceCube::shuffle()
{
    int iterations;
    const int maxIterations = static_cast<const int>(std::pow(n, 3));
    for (iterations = 0; (iterations < maxIterations) || !proper; iterations++)
    {
        if (proper)
        {
            moveFromProper();
        }
        else
        {
            moveFromImproper();
        }
    }

    return iterations;
}

int EfficientIncidenceCube::getEmptySpace(lowerArrayT& arr)
{
    auto predicate = [](int el)->bool{ return el == nullInt; };
    auto it = std::find_if(arr.begin(), arr.end(), predicate);
    if (it == arr.end())
    {
        return -1;
    }
    else
    {
        return it - arr.begin();
    }
}

int EfficientIncidenceCube::coordOf(int x, int y, int z)
{
    auto& array = xyMatrix[x][y];
    std::find(array.begin(), array.end(), z);
    if (std::find(array.begin(), array.end(), z) != array.end())
    {
        return 1;
    }
    else if (std::find(array.begin(), array.end(), minus(z)) != array.end())
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

int EfficientIncidenceCube::plusOneZCoordOf(int x, int y)
{
    return plusOneCoordOf(xyMatrix, x, y);
}

int EfficientIncidenceCube::secondPlusOneZCoordOf(int x, int y)
{
    auto& array = xyMatrix[x][y];
    auto predicate = [](int el)->bool{ return el >= 0; };
    auto it = std::find_if(array.begin(), array.end(), predicate);
    it = std::find_if(it + 1, array.end(), predicate);

    int z = -1;
    if (it != array.end())
        z = it - array.begin();

    if (z >= 0)
        return xyMatrix[x][y][z];
    else
        return -1;
}

int EfficientIncidenceCube::plusOneCoordOf(const matrixT & matrix, int first, int second)
{
    auto& array = matrix[first][second];
    auto predicate = [](int el)->bool{ return el >= 0; };
    auto it = std::find_if(array.begin(), array.end(), predicate);

    int pos = -1;
    if (it != array.end())
        pos = it - array.begin();

    if (pos >= 0)
        return array[pos];
    else
        return -1;
}

int EfficientIncidenceCube::plusOneXCoordOf(int y, int z)
{
    return plusOneCoordOf(yzMatrix, y, z);
}

int EfficientIncidenceCube::plusOneYCoordOf(int x, int z)
{
    return plusOneCoordOf(xzMatrix, x, z);
}

int EfficientIncidenceCube::minusOneCoordOf(int x, int y)
{
    auto& array = xyMatrix[x][y];
    auto predicate = [](int el)->bool{ return el < 0 && el != nullInt; };
    auto it = std::find_if(array.begin(), array.end(), predicate);

    int z = -1;
    if (it != array.end())
        z = it - array.begin();

    if (z >= 0)
        return xyMatrix[x][y][z];
    else
        return -1;
}

void EfficientIncidenceCube::doPlusMinus1Move(orderedTripleT t, int x1, int y1, int z1)
{
    //changes in chosen sub-cube
    //sum 1 to the selected "0" cell

    auto t_x = std::get<0>(t);
    auto t_y = std::get<1>(t);
    auto t_z = std::get<2>(t);

    xyzStore(t_x, t_y, t_z);
    xyzStore(t_x, y1, z1);
    xyzStore(x1, y1, t_z);
    xyzStore(x1, t_y, z1);

    //subtract 1 to the "1" cell
    xyzRemove(t_x, t_y, z1);
    xyzRemove(t_x, y1, t_z);
    xyzRemove(x1, t_y, t_z);
    xyzRemove(x1, y1, z1);
}

void EfficientIncidenceCube::moveFromProper()
{
    orderedTripleT t = select0Cell();
    auto& t_x = std::get<0>(t);
    auto& t_y = std::get<1>(t);
    auto& t_z = std::get<2>(t);

    int x1 = plusOneXCoordOf(t_y, t_z);
    int z1 = plusOneZCoordOf(t_x, t_y);
    int y1 = plusOneYCoordOf(t_x, t_z);

    doPlusMinus1Move(t, x1, y1, z1);

    //check if improper
    //(only one cell can be -1)
    if (coordOf(x1, y1, z1) == -1)
    {
        proper = false;
        improperCell = std::make_tuple(x1, y1, z1);
    }
}

void EfficientIncidenceCube::moveFromImproper()
{
    //get the improper cell:
    auto t = improperCell;
    auto& t_x = std::get<0>(t);
    auto& t_y = std::get<1>(t);
    auto& t_z = std::get<2>(t);

    int x1 = choosePlusOneXCoordOf(t_y, t_z);
    int y1 = choosePlusOneYCoordOf(t_x, t_z);
    int z1 = choosePlusOneZCoordOf(t_x, t_y);

    doPlusMinus1Move(t, x1, y1, z1);

    //this is the only cell that can result -1
    if (coordOf(x1, y1, z1) == -1)
    {
        proper = false;
        improperCell = std::make_tuple(x1, y1, z1);
    }
    else
    {
        proper = true;
        //improperCell = null;
    }
}

int EfficientIncidenceCube::choosePlusOneCoordOf(const matrixT& matrix, int first, int second)
{
    std::uniform_int_distribution<int> u(0, 1);
    bool takeFirst = u(r_engine);

    auto& array = matrix[first][second];
    auto predicate = [](int el)->bool { return (el >= 0); };
    auto it = std::find_if(array.begin(), array.end(), predicate);

    int pos = -1;
    if (it != array.end())
        pos = it - array.begin();

    if (pos == -1)
        return -1;
    if (takeFirst)
        return array[pos];
    else
    {
        it = std::find_if(array.begin() + pos + 1, array.end(), predicate);

        pos = -1;
        if (it != array.end())
            pos = it - array.begin();

        if (pos == -1)
            return -1;
        return array[pos];
    }
}

int EfficientIncidenceCube::choosePlusOneZCoordOf(int x, int y)
{
    return choosePlusOneCoordOf(xyMatrix, x, y);
}

int EfficientIncidenceCube::choosePlusOneXCoordOf(int y, int z)
{
    return choosePlusOneCoordOf(yzMatrix, y, z);
}

int EfficientIncidenceCube::choosePlusOneYCoordOf(int x, int z)
{
    return choosePlusOneCoordOf(xzMatrix, x, z);
}

std::string EfficientIncidenceCube::toString()
{
    if (!proper)
    {
        DEBUG_PRINTLN("Cannot print improper cube");
        return "";
    }

    std::stringstream ss;
    ss << "Incidence cube of size " << n << ":\n";
    for (int x = 0; x < n; x++)
    {
        for (int y = 0; y < n; y++)
        {
            try
            {
                auto elem = plusOneZCoordOf(x, y);
                std::stringstream ss2;
                ss2 << elem;
                auto length = ss2.str().length();

                ss << elem << std::string("    ", length);
            }
            catch (std::out_of_range&)
            {
                ss << "--  ";
            }
        }
        ss << "\n";
    }
    return ss.str();
}

std::string EfficientIncidenceCube::printRaw()
{
    std::stringstream ss;
    ss << "Incidence cube of size " << n << ":\n";
    for (int x = 0; x < n; x++)
    {
        for (int y = 0; y < n; y++)
        {
            auto& matrix = xyMatrix[x][y];
            ss << "["
                << matrix[0]
                << ", "
                << matrix[1]
                << ", "
                << matrix[2]
                << "] ";
        }
        ss << "\n";
    }
    return ss.str();
}

void EfficientIncidenceCube::xyzStore(int x, int y, int z)
{
    add(xyMatrix[x][y], z);
    add(yzMatrix[y][z], x);
    add(xzMatrix[x][z], y);
}

void EfficientIncidenceCube::xyzRemove(int x, int y, int z)
{
    remove(xyMatrix[x][y], z);
    remove(yzMatrix[y][z], x);//if exists, removes
    remove(xzMatrix[x][z], y);
}

orderedTripleT EfficientIncidenceCube::select0Cell()
{
    std::uniform_int_distribution<int> u(0, n - 1);
    int x = u(r_engine);
    int y = u(r_engine);
    int z = u(r_engine);

    while (coordOf(x, y, z) != 0)
    {
        x = u(r_engine);
        y = u(r_engine);
        z = u(r_engine);
    }
    return std::make_tuple(x, y, z);
}

int EfficientIncidenceCube::add(lowerArrayT & arr, int elem)
{
    auto it = std::find(arr.begin(), arr.end(), minus(elem));
    int idx = -1;
    if (it != arr.end())
        idx = it - arr.begin();

    if (idx >= 0)
    { //if -element is found
        arr[idx] = nullInt;//-elem+elem = 0
        return idx;
    }
    else
    {
        idx = getEmptySpace(arr);//look for empty space for the new element
        if (idx == -1)
        {//if full , fail
            return -1;
        }
        else
        {//add the new element
            arr[idx] = elem;
            return idx;//if successful, returns the index of the new element
        }
    }
}

int EfficientIncidenceCube::remove(lowerArrayT & arr, int elem)
{
    auto it = std::find(arr.begin(), arr.end(), elem);
    int idx = -1;
    if (it != arr.end())
        idx = it - arr.begin();

    if (idx >= 0)
    { // if element is found
        arr[idx] = nullInt;
        return idx;
    }
    else
    {//if elem is not found
        idx = getEmptySpace(arr);
        if (idx == -1)
        {//if full, fail
            return -1;
        }
        else
        {
            arr[idx] = minus(elem);//add the negative
            return idx;
        }
    }
}

int EfficientIncidenceCube::minus(int a)
{
    if (a == 0)
        return minus0;
    else if (a == minus0)
        return 0;
    else return (-a);
}
