#pragma once
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace omx_v6 {
using Joints = std::array<double, 4>;
using Observation = std::array<float, 69>;
double now();
struct Config {
  YAML::Node data;
  Joints low{}, high{};
  double period{}, max_cycle{}, velocity{}, acceleration{}, grip_velocity{}, tracking{};
  double joint_step{}, residual_scale{}, training_period{};
  explicit Config(const std::string &yaml, const std::string &urdf);
  void require_commissioned() const;
};
void verify_model(const std::string &directory);
Joints controller_trajectory(const Joints &reference, const Joints &target);
class Actor {
 public:
  explicit Actor(const std::string &directory);
  ~Actor();
  Joints infer(const Observation &observation);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
struct Step {
  Joints q{}, action{}, velocity{};
  double finger{};
};
Step step(const Config &config, const Joints &target, const Joints &goal,
          const Joints &measured, Joints action, const Joints &velocity,
          int phase, int waypoint, int count, double dt,
          double finger, double finger_goal, const Joints &low, const Joints &high);

}  // namespace omx_v6
