#ifndef __INCLUDED_CONSTRAINTS_CUH__
#define __INCLUDED_CONSTRAINTS_CUH__
#include <set>
#include "SquareMatrix.cuh"
#include "Board.cuh"

namespace constraints
{
    typedef std::set<cuda::boardFieldT> constraintsFieldT;
    class Constraints : public cuda::SquareMatrix<constraintsFieldT>
    {
    public:
        Constraints(const cuda::boardFieldT boardSize);
        Constraints(const cuda::Board& board);
        ~Constraints() = default;

        // Output

        void print() const;

    };
}
#endif // !__INCLUDED_CONSTRAINTS_CUH__
