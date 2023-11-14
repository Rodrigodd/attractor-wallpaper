#include "ok_color.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

float frand() {
    static uint32_t seed = 42;
    seed = 1664525 * seed + 1013904223;
    return ((float)seed / UINT32_MAX) * 2.0 - 0.5;
}


auto operator<<(std::ostream& os, ok_color::Lab c) -> std::ostream& {
	return os << std::fixed << std::setprecision(8) << c.L << ' ' << c.a << ' ' << c.b;
}
auto operator<<(std::ostream& os, ok_color::RGB c) -> std::ostream& {
	return os << std::fixed << std::setprecision(8) << c.r << ' ' << c.g << ' ' << c.b;
}
auto operator<<(std::ostream& os, ok_color::HSV c) -> std::ostream& {
	return os << std::fixed << std::setprecision(8) << c.h << ' ' << c.s << ' ' << c.v;
}
auto operator<<(std::ostream& os, ok_color::HSL c) -> std::ostream& {
	return os << std::fixed << std::setprecision(8) << c.h << ' ' << c.s << ' ' << c.l;
}

auto main() -> int {
    for (int i = 0; i < 1000; i ++) {
        std::cout << ok_color::linear_srgb_to_oklab({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::oklab_to_linear_srgb({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::gamut_clip_preserve_chroma({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::gamut_clip_project_to_0_5({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::gamut_clip_project_to_L_cusp({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::gamut_clip_adaptive_L0_0_5({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::gamut_clip_adaptive_L0_L_cusp({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::okhsl_to_srgb({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::srgb_to_okhsl({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::okhsv_to_srgb({ frand(), frand(), frand() }) << '\n';
        std::cout << ok_color::srgb_to_okhsv({ frand(), frand(), frand() }) << '\n';
    }
	return 0;
}

