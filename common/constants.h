#include <array>
#include <vector>

using double2 = std::array<double, 2>;
using double3 = std::array<double, 3>;
using int3 = std::array<int, 3>;

using tensor3d = std::vector< std::vector <std::vector<double> > >;
using tensor2d = std::vector< std::vector<double> >;
using tensor1d = std::vector<double>;

struct buffer3D {
    buffer3D() = default;
    buffer3D(const int3& size): data(tensor1d(size[0] * size[1] * size[2])), size(size) {}
    double get(int x, int y, int z) {
        return data[x * size[1] * size[2] + y * size[2] + z];
    }
    tensor1d data;
    int3 size;
};

namespace consts {
// better names are needed
constexpr double T = 1.0;
constexpr int j0 = 100; // number of iterations (start conditions intializing not included)
constexpr double t = T / j0;

constexpr double l = 1.0;
} // namespace consts
