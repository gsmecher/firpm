#include "firpm.h"
#include "firpm/pmmath.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>

namespace py = pybind11;
using pm::firpm;
using namespace pybind11::literals;

static double to_double(double x) { return x; }
static double to_double(long double x) { return static_cast<double>(x); }
#if HAVE_MPFR
static double to_double(const mpfr::mpreal& x) { return x.toDouble(); }
#endif

/* Custom-band Parks-McClellan: per-band amplitude and weight are arbitrary
 * Python callables evaluated at omega in [0, pi] (FREQ space).  Supports
 * type I (even n) and type II (odd n).  Delegates to pm::firpm(n, fbands)
 * which handles the cos(omega/2) basis change and tap recovery internally. */
template <typename T>
static std::vector<double> firpm_bands_impl(std::size_t n,
        std::vector<std::tuple<double, double, py::function, py::function>> bands,
        double eps, std::size_t nmax, unsigned long prec,
        std::vector<double> /*x0*/)
{
    std::vector<pm::band_t<T>> fbands(bands.size());
    for(std::size_t i{0u}; i < bands.size(); ++i) {
        double f0 = std::get<0>(bands[i]), f1 = std::get<1>(bands[i]);
        py::function amp = std::get<2>(bands[i]);
        py::function wt  = std::get<3>(bands[i]);
        fbands[i].start = pm::pmmath::const_pi<T>() * T(f0);
        fbands[i].stop  = pm::pmmath::const_pi<T>() * T(f1);
        fbands[i].part  = {fbands[i].start, fbands[i].stop};
        fbands[i].space = pm::space_t::FREQ;
        /* exchange() evaluates bands from OpenMP worker threads; take the
         * GIL for each callback.  Amplitude and weight are the true desired
         * values; pm::firpm applies the cos(omega/2) basis change for type II. */
        fbands[i].amplitude = [amp](pm::space_t s, T x) -> T {
            if(s == pm::space_t::CHEBY) x = pm::pmmath::acos(x);
            py::gil_scoped_acquire gil;
            return T(amp(to_double(x)).template cast<double>());
        };
        fbands[i].weight = [wt](pm::space_t s, T x) -> T {
            if(s == pm::space_t::CHEBY) x = pm::pmmath::acos(x);
            py::gil_scoped_acquire gil;
            return T(wt(to_double(x)).template cast<double>());
        };
    }

    pm::pmoutput_t<T> output = [&]() {
        py::gil_scoped_release nogil;
        return pm::firpm<T>(n, fbands, eps, nmax, pm::init_t::UNIFORM, 0u, pm::init_t::UNIFORM, prec);
    }();

    std::vector<double> hd(output.h.size());
    for(std::size_t i{0u}; i < output.h.size(); ++i)
        hd[i] = to_double(output.h[i]);
    return hd;
}

PYBIND11_MODULE(pyfirpm, m){

    py::enum_<pm::init_t>(m, "Strategy")
        .value("UNIFORM", pm::init_t::UNIFORM)
        .value("SCALING", pm::init_t::SCALING)
        .value("AFP", pm::init_t::AFP)
        .export_values();

    m.def("firpm", [](std::size_t n,
              std::vector<double> f, std::vector<double> a, std::vector<double> w,
              double eps,
              std::size_t nmax,
              pm::init_t strategy,
              std::size_t depth,
              pm::init_t rstrategy,
              unsigned long prec) {
            return firpm<double>(n, f, a, w, eps, nmax, strategy, depth, rstrategy, prec).h;
        },
        "n"_a,
        "f"_a,
        "a"_a,
        "w"_a,
        "eps"_a=0.01,
        "nmax"_a=4,
        "strategy"_a=pm::init_t::UNIFORM,
        "depth"_a=0u,
        "rstrategy"_a=pm::init_t::UNIFORM,
        "prec"_a=165ul,
        "Parks-McClellan routine for implementing type I and II FIR filters.");

    m.def("firpm_bands", &firpm_bands_impl<double>,
        "n"_a, "bands"_a, "eps"_a=0.01, "nmax"_a=4, "prec"_a=165ul,
        "x0"_a=std::vector<double>{},
        "Parks-McClellan with per-band callable amplitude/weight (type I/II, "
        "double precision).  Returns tap vector h.");

#if HAVE_MPFR
    m.def("firpm_bands_mp", [](std::size_t n,
              std::vector<std::tuple<double, double, py::function, py::function>> bands,
              double eps, std::size_t nmax, unsigned long prec,
              std::vector<double> x0) {
            mpfr::mpreal::set_default_prec(prec);
            return firpm_bands_impl<mpfr::mpreal>(n, bands, eps, nmax, prec, x0);
        },
        "n"_a, "bands"_a, "eps"_a=0.01, "nmax"_a=4, "prec"_a=165ul,
        "x0"_a=std::vector<double>{},
        "As firpm_bands (type I/II), computed in MPFR arbitrary precision (prec bits); "
        "taps extracted before rounding to double.  Returns tap vector h.");
#endif
}
