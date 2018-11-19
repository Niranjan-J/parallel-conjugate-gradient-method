#include "linearalg.hpp"
#include "testing.hpp"
std::string GetCurrentTimeForFileName()
{
    auto time = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%F_%T"); // ISO 8601 without timezone information.
    auto s = ss.str();
    std::replace(s.begin(), s.end(), ':', '-');
    return s;
}

int main()
{
    // auto [avg, std_dev] = tester(6000, true, 4, 100, false);
    // cout << avg << " " << std_dev << endl;
    // tie(avg, std_dev) = tester(3, false);
    // cout << avg << " " << std_dev << endl;

    std::ofstream output("test_results" + GetCurrentTimeForFileName() + ".txt");

    // ofstream output("test_results.txt");

    for (int psd = 0; psd < 2; psd++)
    {
        for (int num_threads = 1; num_threads < 5; num_threads++)
        {
            for (int size = 100; size <= 1000; size += 100)
            {
                auto [avg, std_dev] = tester(size, psd, num_threads);
                output << psd << " " << num_threads << " " << size << " " << avg << " " << std_dev << endl;
                cout << psd << " " << num_threads << " " << size << " " << avg << " " << std_dev << endl;
            }
        }
    }

    return 0;
}
