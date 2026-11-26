// onnx_probe_cpp <model_dir>: checks the actor against the frozen parity vectors
// (max error 1e-5); the pick-and-place launch starts nothing else until it passes.

#include "omx_turtlebot3_control_cpp/control.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: onnx_probe_cpp MODEL_DIRECTORY");
    omx_v6::Actor actor(argv[1]);
    std::ifstream f(std::string(argv[1])+"/parity_vectors.txt");
    std::string magic; int count = 0;
    if (!(f >> magic >> count) || magic != "OMXV6TEST01" || count < 1 || count > 100000) throw std::runtime_error("Invalid parity vectors");
    double error = 0; omx_v6::Observation obs;
    for (int k = 0; k < count; ++k) {
      for (auto &x : obs) if (!(f >> x) || !std::isfinite(x)) throw std::runtime_error("Invalid observation");
      const auto actual = actor.infer(obs);
      for (double x : actual) { double expected; if (!(f >> expected) || !std::isfinite(expected)) throw std::runtime_error("Invalid expected output"); error = std::max(error, std::abs(x-expected)); }
    }
    std::string extra; if (f >> extra) throw std::runtime_error("Trailing parity data");
    if (error > 1e-5) throw std::runtime_error("ONNX parity failed: max_abs_error="+std::to_string(error));
    const double start = omx_v6::now();
    for (int i = 0; i < 2000; ++i) actor.infer(obs);
    std::cout << "{\"backend\":\"cpp_opencv_dnn_onnx_cpu\",\"samples\":" << count << ",\"max_abs_error\":" << error << ",\"mean_inference_us\":" << (omx_v6::now()-start)*1e6/2000 << ",\"motor_writes\":0}\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
