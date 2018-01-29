#pragma once

namespace AMP
{
    template<typename Left, typename Right> class Pair
    {
    public:
        Pair() restrict(amp);
        //Pair(const Pair<Left, Right> & p) restrict(amp) = default;
        Pair(Left l, Right r) restrict(amp);
        ~Pair() restrict(amp);

        Left first;
        Right second;
    };

    template<typename Left, typename Right>
    inline Pair<Left, Right>::Pair() restrict(amp)
    {
        // Nothing to do
    }

    template<typename Left, typename Right>
    inline Pair<Left, Right>::Pair(Left l, Right r) restrict(amp) : first(l), second(r)
    {
        // Nothing to do
    }

    template<typename Left, typename Right>
    inline Pair<Left, Right>::~Pair() restrict(amp)
    {
        // Nothing to do
    }
}
