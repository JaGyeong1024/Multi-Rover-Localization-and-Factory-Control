// v6 actor: model directory checks (SHA256SUMS, contract), OpenCV DNN inference of the
// original ONNX with the final Clip applied natively, and the residual step formula.

#include "omx_turtlebot3_control_cpp/control.hpp"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <openssl/evp.h>
#include <urdf/model.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace omx_v6 {
namespace fs = std::filesystem;
namespace {
void require(bool ok, const std::string &reason) { if (!ok) throw std::runtime_error(reason); }
double number(const YAML::Node &n) {
  const double v = n.as<double>();
  require(std::isfinite(v), "Nonfinite configuration");
  return v;
}
std::string sha256(const fs::path &p) {
  std::ifstream file(p, std::ios::binary);
  require(file.good(), "Cannot read checksum input: " + p.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "SHA256 init failed");
  char bytes[65536];
  while (file.read(bytes, sizeof(bytes)) || file.gcount())
    require(EVP_DigestUpdate(ctx.get(), bytes, file.gcount()) == 1, "SHA256 update failed");
  require(file.eof(), "Checksum input read failed");
  unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int size = 0;
  require(EVP_DigestFinal_ex(ctx.get(), digest, &size) == 1, "SHA256 final failed");
  std::ostringstream out;
  for (unsigned i = 0; i < size; ++i) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(digest[i]);
  return out.str();
}
}
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
Config::Config(const std::string &yaml, const std::string &robot) : data(YAML::LoadFile(yaml)) {
  require(data["schema_version"].as<int>() == 1 && data["model_version"].as<std::string>() == "omx_turtlebot3_v6", "Wrong runtime contract");
  const auto c = data["control"];
  period = number(c["period_s"]); max_cycle = number(c["max_cycle_s"]);
  velocity = number(c["joint_velocity_rad_s"]); acceleration = number(c["joint_acceleration_rad_s2"]);
  grip_velocity = number(c["gripper_velocity_m_s"]); tracking = number(c["max_tracking_error_rad"]);
  require(period > 0 && period <= max_cycle && max_cycle <= .2 && velocity > 0 && velocity <= .72 && acceleration > 0 && grip_velocity > 0 && grip_velocity <= .02 && tracking > 0, "Invalid control envelope");
  require(c["command_route"].as<std::string>() == "mp_control_equivalent", "C++ transport requires explicit mp_control_equivalent route");
  const auto s = data["serial"];
  require(s["id"].as<int>() == 200 && s["model_number"].as<int>() == 20480 && s["baudrate"].as<int>() == 1000000, "Unexpected OpenCR contract");
  require(s["joint_signs"].size() == 4 && s["joint_zero_ticks"].size() == 4, "Invalid joint mapping length");
  for (size_t i = 0; i < 4; ++i) {
    require(std::abs(number(s["joint_signs"][i])) == 1, "Invalid calibrated joint sign");
    const int zero = s["joint_zero_ticks"][i].as<int>();
    require(zero >= 0 && zero <= 4096, "Invalid joint zero");
  }
  require(number(s["gripper_m_per_rad"]) == -.015, "Unexpected gripper conversion");
  require(sha256(robot) == data["vision"]["robot_urdf_sha256"].as<std::string>(), "Robot URDF checksum mismatch");
  urdf::Model model;
  require(model.initFile(robot), "Cannot parse robot URDF");
  for (int i = 0; i < 4; ++i) {
    auto joint = model.getJoint("joint" + std::to_string(i+1));
    require(joint && joint->limits && joint->type == urdf::Joint::REVOLUTE, "Invalid URDF joint");
    low[i] = joint->limits->lower; high[i] = joint->limits->upper;
  }
  // Immutable v6 formula; assert its training config again when loading the actor.
  joint_step = .012; residual_scale = .12; training_period = .02;
}
void Config::require_commissioned() const {
  const auto c = data["commissioning"];
  for (const auto *key : {"joint_mapping_verified", "camera_extrinsics_verified", "marker_mount_verified", "scene_alignment_verified", "grasp_thresholds_verified", "placement_thresholds_verified", "hardware_watchdog_verified"})
    require(c[key].as<bool>(), std::string("Commissioning incomplete: ") + key);
  require(!c["evidence_directory"].as<std::string>().empty(), "Commissioning evidence missing");
  const auto e = data["estimation"];
  const double idle = number(e["current_idle_raw"]), grip = number(e["current_grip_delta_raw"]), release = number(e["current_release_delta_raw"]);
  require(std::isfinite(idle) && release >= 0 && grip > release, "Invalid physical current thresholds");
  require(!data["vision"]["camera_mount_candidate"].as<bool>(false), "Unmeasured camera mount");
  require(data["vision"]["marker_size_confirmed"].as<bool>(), "Marker dimensions unverified");
}
void verify_model(const std::string &directory) {
  const auto root = fs::canonical(directory);
  std::ifstream sums(root / "SHA256SUMS");
  require(sums.good(), "Missing model checksum manifest");
  std::string line; bool actor = false, manifest = false, training = false;
  while (std::getline(sums, line)) {
    require(line.size() > 66 && line.substr(64, 2) == "  ", "Malformed SHA256SUMS");
    const auto name = line.substr(66);
    require(fs::path(name).filename() == name && name != "." && name != "..", "Unsafe model checksum path");
    require(fs::canonical(root / name).parent_path() == root, "Model symlink outside directory");
    require(sha256(root / name) == line.substr(0, 64), "Model checksum mismatch: " + name);
    actor |= name == "actor.onnx"; manifest |= name == "manifest.json"; training |= name == "training_config.yaml";
  }
  require(actor && manifest && training, "Model checksum entries missing");
  const auto m = YAML::LoadFile((root / "manifest.json").string());
  require(m["model_version"].as<std::string>() == "omx_turtlebot3_v6" && m["observation_size"].as<int>() == 69 && m["action_size"].as<int>() == 4 && m["action_semantics"].as<std::string>() == "bounded_joint_residual_not_absolute_angles", "Wrong ONNX model contract");
  require(m["onnx"]["input"].as<std::string>() == "observation" && m["onnx"]["output"].as<std::string>() == "action", "Wrong ONNX tensor names");
  const auto t = YAML::LoadFile((root / "training_config.yaml").string());
  require(number(t["joint_step"]) == .012 && number(t["residual_scale"]) == .12 && std::abs(number(t["physics_timestep"])*number(t["frame_skip"])-.02) < 1e-12, "Unsupported training control formula");
}
struct Actor::Impl { cv::dnn::Net net; std::string projection; std::mutex mutex; };
Actor::Actor(const std::string &directory) : impl_(std::make_unique<Impl>()) {
  verify_model(directory);
  cv::setNumThreads(1);
  impl_->net = cv::dnn::readNetFromONNX(directory + "/actor.onnx");
  require(!impl_->net.empty(), "ONNX network is empty");
  impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  // This immutable v6 ONNX ends with Clip(l3, -1, 1). OpenCV 4.5.4 on
  // Jetson imports Clip as ReLU6 and incorrectly zeroes negative values.
  // Execute the original ONNX projection and apply exactly that final Clip
  // below, without rewriting the model or switching inference backends.
  require(sha256(fs::path(directory)/"actor.onnx") ==
    "fe3656701f104022872ad804ac120496c4b31de8cf9e7c3a2f94250e4ceef00c",
    "Unsupported v6 ONNX graph; review projection/Clip contract");
  for (const auto &name : impl_->net.getLayerNames()) {
    if (name == "l3" || (name.size() > 3 && name.substr(name.size()-3) == "!l3")) {
      require(impl_->projection.empty(), "Ambiguous ONNX projection");
      const auto layer = impl_->net.getLayer(impl_->net.getLayerId(name));
      require(layer && (layer->type == "InnerProduct" || layer->type == "Gemm"), "Invalid ONNX projection layer");
      impl_->projection = name;
    }
  }
  require(!impl_->projection.empty(), "Missing immutable ONNX l3 projection");
  impl_->net.enableFusion(false);
  infer(Observation{});  // Warm up before control timing starts; never touches hardware.
  // A successful load is insufficient: require the frozen conformance vectors
  // before any caller can open/arm a hardware transport.
  std::ifstream fixture(fs::path(directory)/"parity_vectors.txt");
  std::string magic; int count = 0;
  require(bool(fixture >> magic >> count) && magic == "OMXV6TEST01" && count == 973, "Invalid startup parity fixture");
  double maximum_error = 0;
  for (int i = 0; i < count; ++i) {
    Observation obs;
    for (auto &x : obs) require(bool(fixture >> x) && std::isfinite(x), "Invalid startup parity observation");
    const auto actual = infer(obs);
    for (double x : actual) {
      double expected;
      require(bool(fixture >> expected) && std::isfinite(expected), "Invalid startup parity action");
      maximum_error = std::max(maximum_error, std::abs(x-expected));
    }
  }
  std::string extra;
  require(!(fixture >> extra), "Trailing startup parity data");
  require(maximum_error <= 1e-5, "Native ONNX startup parity failed: " + std::to_string(maximum_error));
}
Actor::~Actor() = default;
Joints Actor::infer(const Observation &observation) {
  for (float x : observation) require(std::isfinite(x), "Nonfinite 69D observation");
  std::lock_guard<std::mutex> guard(impl_->mutex);
  cv::Mat input(1, 69, CV_32F, const_cast<float *>(observation.data()));
  impl_->net.setInput(input, "observation");
  const cv::Mat output = impl_->net.forward(impl_->projection);
  require(output.type() == CV_32F && output.total() == 4 && output.isContinuous(), "Invalid ONNX output shape/type");
  Joints action;
  for (size_t i = 0; i < 4; ++i) {
    const float x = output.ptr<float>()[i];
    require(std::isfinite(x), "Nonfinite ONNX output");
    action[i] = std::clamp(double(x), -1., 1.);
  }
  return action;
}
Joints controller_trajectory(const Joints &reference, const Joints &target) {
  for (size_t i = 0; i < 4; ++i) require(std::isfinite(reference[i]) && std::isfinite(target[i]), "Nonfinite trajectory");
  require(std::abs(target[2]-reference[2]) <= .20+1e-12, "mp_control joint3 delta exceeded");
  const double raw = 2*reference[2]-target[2];
  for (double x : {reference[2], raw, target[2]}) require(x >= -2.70 && x <= 1.38, "mp_control joint3 bounds exceeded");
  Joints output = target;
  output[2] = 2*reference[2]-raw;
  // Same captured reference for both transforms; joint4 already is the final angle.
  return output;
}
Step step(const Config &c, const Joints &target, const Joints &goal,
          const Joints &measured, Joints action, const Joints &velocity,
          int phase, int waypoint, int count, double dt, double finger,
          double finger_goal, const Joints &low, const Joints &high) {
  require(std::isfinite(dt) && dt > 0 && dt <= c.max_cycle, "Control deadline exceeded");
  require(phase >= 0 && phase < 9 && count > 0 && waypoint >= 0 && waypoint < count, "Invalid phase/waypoint");
  require(std::isfinite(finger) && std::isfinite(finger_goal) && finger >= -.010 && finger <= .019 && finger_goal >= -.010 && finger_goal <= .019, "Invalid gripper target");
  double speed = 1.;
  if (phase == 0 || phase == 5 || (phase == 2 && waypoint < count-2) || (phase == 4 && waypoint > 0) || (phase == 6 && waypoint == 0) || (phase == 8 && waypoint > 0)) speed = 1.2;
  double distance = 0;
  for (size_t i = 0; i < 4; ++i) {
    for (double x : {target[i], goal[i], measured[i], action[i], velocity[i], low[i], high[i]}) require(std::isfinite(x), "Nonfinite control step");
    require(low[i] < high[i] && target[i] >= std::max(low[i], c.low[i]) && target[i] <= std::min(high[i], c.high[i]) && goal[i] >= std::max(low[i], c.low[i]) && goal[i] <= std::min(high[i], c.high[i]), "Control target outside URDF/planner limits");
    distance = std::max(distance, std::abs(goal[i]-target[i]));
    action[i] = std::clamp(action[i], -1., 1.);
  }
  if (phase == 5) {
    action[1] = action[2] = action[3] = 0;
    if (std::abs(measured[0]+std::acos(-1.)/2) < .03) action[0] = 0;
  }
  const double scale = c.joint_step*speed;
  Step result; result.action = action;
  for (size_t i = 0; i < 4; ++i) {
    const double reference = (goal[i]-target[i])/std::max(scale, distance);
    double delta = scale*std::clamp(reference+c.residual_scale/speed*action[i], -1., 1.)*dt/c.training_period;
    if (phase == 5 && i > 0) delta = 0;
    const double requested = std::clamp(delta/dt, -c.velocity, c.velocity);
    result.velocity[i] = velocity[i]+std::clamp(requested-velocity[i], -c.acceleration*dt, c.acceleration*dt);
    result.q[i] = std::clamp(target[i]+result.velocity[i]*dt, std::max(low[i], c.low[i]), std::min(high[i], c.high[i]));
    if (phase == 5 && i > 0) { result.q[i] = target[i]; result.velocity[i] = 0; }
  }
  result.finger = finger+std::clamp(finger_goal-finger, -c.grip_velocity*dt, c.grip_velocity*dt);
  return result;
}
}  // namespace omx_v6
