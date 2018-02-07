#pragma once

namespace cuda
{
    template<typename T> class PointerIterator
    {
    public:
        PointerIterator(T* pointer);
        ~PointerIterator() = default;

        PointerIterator& operator++ ();
        PointerIterator operator++ (int);
    private:
        T * ptr;
    };

    template<typename T>
    inline PointerIterator<T>::PointerIterator(T * pointer) : ptr(pointer)
    {
        // Nothing to do
    }

    template<typename T>
    inline PointerIterator & PointerIterator<T>::operator++()
    {
        ++ptr;
        return *this;
    }

    template<typename T>
    inline PointerIterator PointerIterator<T>::operator++(int)
    {
        return PointerIterator(ptr++);
    }
}
