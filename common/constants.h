#include <array>
#include <vector>

using double2 = std::array<double, 2>;
using double3 = std::array<double, 3>;

using tensor3d = std::vector< std::vector <std::vector<double> > >;
using tensor2d = std::vector< std::vector<double> >;
using tensor1d = std::vector<double>;

namespace consts {
// better names are needed
constexpr double T = 1.0;
constexpr int j0 = 100; // number of iterations (start conditions intializing not included)
constexpr double t = T / j0;

constexpr double l = 1.0;
} // namespace consts
