#include "linearalg.hpp"
#include "testing.hpp"

int main()
{
    auto [avg, std_dev] = tester(3, true);
    cout << avg << " " << std_dev << endl;
    tie(avg, std_dev) = tester(3, false);
    cout << avg << " " << std_dev << endl;

    return 0;
}
