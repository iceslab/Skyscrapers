#include "AMPUtilities.h"

namespace AMP
{
    void AMPUtilities::listAllAccelerators()
    {
        std::vector<Concurrency::accelerator> accs = Concurrency::accelerator::get_all();
        for (size_t i = 0; i < accs.size(); i++)
        {
            std::wcout << "Device no. " << i << ": " << "\n";
            std::wcout << "Description: " << accs[i].description << "\n";
            std::wcout << "Path: " << accs[i].device_path << "\n";
            std::wcout << "Memory: " << accs[i].dedicated_memory << " KB\n";
            std::wcout << "CPU shared memory: " << boolToString(accs[i].supports_cpu_shared_memory) << "\n";
            std::wcout << "Double precision: " << boolToString(accs[i].supports_double_precision) << "\n";
            std::wcout << "Limited double precision:  " << boolToString(accs[i].supports_limited_double_precision) << "\n\n";
        }
    }
}
