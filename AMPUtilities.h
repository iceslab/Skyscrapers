#pragma once
#include <amp.h>
#include <iostream>
#include <vector>

namespace utils
{
    class AMPUtilities
    {
    public:
        AMPUtilities() = delete;
        ~AMPUtilities() = delete;

        static void listAllAccelerators();

        static inline auto boolToString(bool val)
        {
            return val ? "true" : "false";
        }
    };
}
