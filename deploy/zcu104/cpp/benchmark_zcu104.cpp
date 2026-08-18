#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct CpuFlatTensorBuffer : public vart::TensorBuffer {
  CpuFlatTensorBuffer(void* data, const xir::Tensor* tensor)
      : vart::TensorBuffer{tensor}, data_{data} {}

  std::pair<uint64_t, size_t> data(const std::vector<int> idx = {}) override {
    uint32_t element_bytes =
        static_cast<uint32_t>(std::ceil(tensor_->get_data_type().bit_width / 8.0f));
    if (idx.empty()) {
      return {reinterpret_cast<uint64_t>(data_),
              static_cast<size_t>(tensor_->get_element_num()) * element_bytes};
    }

    const auto dims = tensor_->get_shape();
    if (idx.size() != dims.size()) {
      throw std::runtime_error("TensorBuffer index rank mismatch");
    }

    size_t offset = 0;
    for (size_t k = 0; k < dims.size(); ++k) {
      size_t stride = 1;
      for (size_t m = k + 1; m < dims.size(); ++m) {
        stride *= static_cast<size_t>(dims[m]);
      }
      offset += static_cast<size_t>(idx[k]) * stride;
    }

    const size_t elem_num = static_cast<size_t>(tensor_->get_element_num());
    return {reinterpret_cast<uint64_t>(data_) + offset * element_bytes,
            (elem_num - offset) * element_bytes};
  }

 private:
  void* data_;
};

struct Options {
  std::string model = "../model/trafficsignnet_int8.xmodel";
  std::string dataset = "../dataset";
  std::string output_dir = "../results_cpp";
  int warmup = 1000;
  int runs = 5;
  double seconds = 60.0;
  int workload_size = 100;
  int workload_runs = 10;
  int seed = 42;
  int max_workers = 4;
  int sweep_runs = 3;
  double sweep_seconds = 10.0;
  int sweep_warmup = 100;
  int parallel_workers = 2;
  int cold_runs = 3;
  int full_e2e_runs = 3;
  int power_sample_ms = 20;
  bool skip_full_test = false;
  bool require_full_gtsrb_test = false;
  bool skip_cold = false;
  bool skip_max_confirmation = false;
  bool cold_child = false;
  std::string cold_image;
};

struct ImageRef {
  fs::path path;
  int label = -1;
};

struct MemoryImage {
  fs::path path;
  int label = -1;
  std::vector<uint8_t> rgb;
};

struct PowerStats {
  bool available = false;
  std::string sensor_name;
  std::string sensor_path;
  uint64_t samples = 0;
  double duration_s = 0.0;
  double mean_w = 0.0;
  double min_w = 0.0;
  double max_w = 0.0;
  double std_w = 0.0;
  double ci95_low_w = 0.0;
  double ci95_high_w = 0.0;
  double energy_j = 0.0;
};

struct LatencyStats {
  uint64_t count = 0;
  double elapsed_s = 0.0;
  double fps = 0.0;
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double std_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double p90_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double p999_ms = 0.0;
  PowerStats power;
};

struct WorkloadRun {
  uint64_t images = 0;
  uint64_t correct = 0;
  double accuracy = 0.0;
  LatencyStats perf;
};

struct SweepPoint {
  int workers = 0;
  std::vector<LatencyStats> runs;
};

struct WorkloadSweepPoint {
  int workers = 0;
  std::vector<WorkloadRun> runs;
};

struct BenchmarkResults {
  std::vector<LatencyStats> single_model_only;
  std::vector<LatencyStats> single_host_to_host;
  std::vector<WorkloadRun> fixed_memory_single;
  std::vector<WorkloadRun> fixed_end_to_end_single;
  WorkloadRun full_test_end_to_end;
  bool full_test_ran = false;
  std::vector<SweepPoint> sweep_model_only;
  std::vector<SweepPoint> sweep_host_to_host;
  int best_model_workers = 1;
  int best_host_workers = 1;
  std::vector<LatencyStats> max_model_only;
  std::vector<LatencyStats> max_host_to_host;
  std::vector<WorkloadRun> fixed_memory_parallel;
  std::vector<WorkloadSweepPoint> full_e2e_worker_sweep;
  int best_full_e2e_workers = 1;
  std::vector<double> cold_start_ms;
};

static void print_usage(const char* exe) {
  std::cout
      << "Usage: " << exe << " [options]\n\n"
      << "Main options:\n"
      << "  --model PATH                 XMODEL path\n"
      << "  --dataset PATH               Dataset root (class folders 00000..00042)\n"
      << "  --output-dir PATH            Output directory\n"
      << "  --warmup N                   Single-stream warmup (default 1000)\n"
      << "  --runs N                     Sustained runs (default 5)\n"
      << "  --seconds S                  Seconds per sustained run (default 60)\n"
      << "  --workload-size N            Fixed workload images (default 100)\n"
      << "  --workload-runs N            Fixed workload repetitions (default 10)\n"
      << "  --max-workers N              Worker sweep 1..N (default 4)\n"
      << "  --sweep-runs N               Runs per worker count (default 3)\n"
      << "  --sweep-seconds S            Seconds per sweep run (default 10)\n"
      << "  --sweep-warmup N             Warmup per worker (default 100)\n"
      << "  --parallel-workers N         Initial parallel fixed workload workers (default 2)\n"
      << "  --cold-runs N                Fresh-process cold starts (default 3)\n"
      << "  --full-e2e-runs N            Full-dataset end-to-end repetitions per worker count (default 3)\n"
      << "  --power-sample-ms N          INA226 power sampling interval in ms (default 20)\n"
      << "  --seed N                     Deterministic workload seed (default 42)\n"
      << "  --skip-full-test             Skip all-images end-to-end test\n"
      << "  --require-full-gtsrb-test    Require exactly 12,630 images spanning labels 0..42\n"
      << "  --skip-cold                  Skip fresh-process cold-start test\n"
      << "  --skip-max-confirmation      Skip long best-worker 5x60s confirmation\n"
      << "  --help                       Show this help\n";
}

static Options parse_args(int argc, char** argv) {
  Options o;
  auto require_value = [&](int& i, const std::string& flag) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
    return argv[++i];
  };

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (a == "--model") {
      o.model = require_value(i, a);
    } else if (a == "--dataset") {
      o.dataset = require_value(i, a);
    } else if (a == "--output-dir") {
      o.output_dir = require_value(i, a);
    } else if (a == "--warmup") {
      o.warmup = std::stoi(require_value(i, a));
    } else if (a == "--runs") {
      o.runs = std::stoi(require_value(i, a));
    } else if (a == "--seconds") {
      o.seconds = std::stod(require_value(i, a));
    } else if (a == "--workload-size") {
      o.workload_size = std::stoi(require_value(i, a));
    } else if (a == "--workload-runs") {
      o.workload_runs = std::stoi(require_value(i, a));
    } else if (a == "--seed") {
      o.seed = std::stoi(require_value(i, a));
    } else if (a == "--max-workers") {
      o.max_workers = std::stoi(require_value(i, a));
    } else if (a == "--sweep-runs") {
      o.sweep_runs = std::stoi(require_value(i, a));
    } else if (a == "--sweep-seconds") {
      o.sweep_seconds = std::stod(require_value(i, a));
    } else if (a == "--sweep-warmup") {
      o.sweep_warmup = std::stoi(require_value(i, a));
    } else if (a == "--parallel-workers") {
      o.parallel_workers = std::stoi(require_value(i, a));
    } else if (a == "--cold-runs") {
      o.cold_runs = std::stoi(require_value(i, a));
    } else if (a == "--full-e2e-runs") {
      o.full_e2e_runs = std::stoi(require_value(i, a));
    } else if (a == "--power-sample-ms") {
      o.power_sample_ms = std::stoi(require_value(i, a));
    } else if (a == "--skip-full-test") {
      o.skip_full_test = true;
    } else if (a == "--require-full-gtsrb-test") {
      o.require_full_gtsrb_test = true;
    } else if (a == "--skip-cold") {
      o.skip_cold = true;
    } else if (a == "--skip-max-confirmation") {
      o.skip_max_confirmation = true;
    } else if (a == "--cold-child") {
      o.cold_child = true;
    } else if (a == "--cold-image") {
      o.cold_image = require_value(i, a);
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }

  if (o.warmup < 0 || o.runs <= 0 || o.seconds <= 0.0 ||
      o.workload_size <= 0 || o.workload_runs <= 0 || o.max_workers <= 0 ||
      o.sweep_runs <= 0 || o.sweep_seconds <= 0.0 || o.sweep_warmup < 0 ||
      o.parallel_workers <= 0 || o.cold_runs <= 0 || o.full_e2e_runs <= 0 ||
      o.power_sample_ms <= 0) {
    throw std::runtime_error("Benchmark numeric arguments must be positive (warmups may be zero)");
  }
  return o;
}


static std::string read_first_line(const fs::path& path) {
  std::ifstream is(path);
  std::string value;
  if (is) std::getline(is, value);
  return value;
}

static fs::path detect_power_sensor_path() {
  const fs::path root("/sys/class/hwmon");
  if (!fs::exists(root)) return {};
  fs::path fallback;
  std::error_code ec;
  for (const auto& hw : fs::directory_iterator(root, ec)) {
    if (ec || !hw.is_directory()) continue;
    const std::string name = read_first_line(hw.path() / "name");
    for (const auto& entry : fs::directory_iterator(hw.path(), ec)) {
      if (ec || !entry.is_regular_file()) continue;
      const std::string fn = entry.path().filename().string();
      if (fn.rfind("power", 0) == 0 && fn.size() > 6 &&
          fn.substr(fn.size() - 6) == "_input") {
        if (fallback.empty()) fallback = entry.path();
        if (name == "ina226") return entry.path();
      }
    }
  }
  return fallback;
}

static double read_power_watts(const fs::path& path) {
  std::ifstream is(path);
  long long microwatts = 0;
  if (!(is >> microwatts)) return std::numeric_limits<double>::quiet_NaN();
  return static_cast<double>(microwatts) / 1.0e6;
}

class PowerSampler {
 public:
  explicit PowerSampler(int interval_ms)
      : interval_ms_(std::max(1, interval_ms)), sensor_path_(detect_power_sensor_path()) {
    if (!sensor_path_.empty()) {
      sensor_name_ = read_first_line(sensor_path_.parent_path() / "name");
    }
  }

  bool available() const { return !sensor_path_.empty(); }
  const fs::path& sensor_path() const { return sensor_path_; }
  const std::string& sensor_name() const { return sensor_name_; }

  void start() {
    samples_.clear();
    if (!available()) return;
    running_.store(true, std::memory_order_release);
    start_ = Clock::now();
    worker_ = std::thread([this]() {
      auto next = Clock::now();
      while (running_.load(std::memory_order_acquire)) {
        const double w = read_power_watts(sensor_path_);
        if (std::isfinite(w)) samples_.push_back(w);
        next += std::chrono::milliseconds(interval_ms_);
        std::this_thread::sleep_until(next);
      }
    });
  }

  PowerStats stop() {
    PowerStats out;
    if (!available()) return out;
    const auto stop_time = Clock::now();
    running_.store(false, std::memory_order_release);
    if (worker_.joinable()) worker_.join();
    const double duration_s = std::chrono::duration<double>(stop_time - start_).count();
    out.sensor_name = sensor_name_;
    out.sensor_path = sensor_path_.string();
    out.duration_s = duration_s;
    out.samples = samples_.size();
    if (samples_.empty()) return out;
    out.available = true;
    out.min_w = *std::min_element(samples_.begin(), samples_.end());
    out.max_w = *std::max_element(samples_.begin(), samples_.end());
    out.mean_w = std::accumulate(samples_.begin(), samples_.end(), 0.0) /
                 static_cast<double>(samples_.size());
    if (samples_.size() > 1) {
      double sq = 0.0;
      for (double x : samples_) {
        const double d = x - out.mean_w;
        sq += d * d;
      }
      out.std_w = std::sqrt(sq / static_cast<double>(samples_.size() - 1));
      const double half = 1.96 * out.std_w / std::sqrt(static_cast<double>(samples_.size()));
      out.ci95_low_w = out.mean_w - half;
      out.ci95_high_w = out.mean_w + half;
    } else {
      out.ci95_low_w = out.mean_w;
      out.ci95_high_w = out.mean_w;
    }
    out.energy_j = out.mean_w * duration_s;
    return out;
  }

 private:
  int interval_ms_ = 20;
  fs::path sensor_path_;
  std::string sensor_name_;
  std::atomic<bool> running_{false};
  Clock::time_point start_{};
  std::thread worker_;
  std::vector<double> samples_;
};

static std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

static bool is_image_path(const fs::path& p) {
  const std::string e = lower(p.extension().string());
  return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".ppm" || e == ".bmp";
}

static int parse_label_from_parent(const fs::path& p) {
  const std::string name = p.parent_path().filename().string();
  try {
    size_t consumed = 0;
    int value = std::stoi(name, &consumed);
    if (consumed != name.size()) return -1;
    return value;
  } catch (...) {
    return -1;
  }
}

static std::vector<ImageRef> list_dataset(const fs::path& root) {
  if (!fs::exists(root)) throw std::runtime_error("Dataset not found: " + root.string());
  std::vector<ImageRef> out;
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) continue;
    if (!is_image_path(entry.path())) continue;
    int label = parse_label_from_parent(entry.path());
    if (label < 0) continue;
    out.push_back({entry.path(), label});
  }
  std::sort(out.begin(), out.end(), [](const ImageRef& a, const ImageRef& b) {
    return a.path.string() < b.path.string();
  });
  if (out.empty()) throw std::runtime_error("No labeled images found under: " + root.string());
  return out;
}

static std::vector<ImageRef> select_workload(std::vector<ImageRef> all, int n, int seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::shuffle(all.begin(), all.end(), rng);
  if (static_cast<size_t>(n) < all.size()) all.resize(static_cast<size_t>(n));
  return all;
}

static std::vector<uint8_t> load_rgb_resized(const fs::path& path, int width, int height) {
  cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (bgr.empty()) throw std::runtime_error("Failed to read image: " + path.string());

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  if (rgb.cols != width || rgb.rows != height) {
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    rgb = resized;
  }
  if (!rgb.isContinuous()) rgb = rgb.clone();

  const size_t bytes = static_cast<size_t>(width) * height * 3;
  return std::vector<uint8_t>(rgb.data, rgb.data + bytes);
}

static std::vector<MemoryImage> preload_images(const std::vector<ImageRef>& refs,
                                                int width, int height) {
  std::vector<MemoryImage> out;
  out.reserve(refs.size());
  for (const auto& r : refs) {
    out.push_back({r.path, r.label, load_rgb_resized(r.path, width, height)});
  }
  return out;
}

static const xir::Subgraph* find_single_dpu_subgraph(const xir::Graph* graph) {
  auto root = graph->get_root_subgraph();
  auto children = root->children_topological_sort();
  std::vector<const xir::Subgraph*> dpu;
  for (auto* c : children) {
    if (!c->has_attr("device")) continue;
    if (c->get_attr<std::string>("device") == "DPU") dpu.push_back(c);
  }
  if (dpu.size() != 1) {
    std::ostringstream oss;
    oss << "Expected exactly one DPU subgraph, found " << dpu.size();
    throw std::runtime_error(oss.str());
  }
  return dpu[0];
}

class QuantizerLUT {
 public:
  explicit QuantizerLUT(int fix_point) : fix_point_(fix_point) {
    const float scale = std::exp2f(static_cast<float>(fix_point_));
    for (int i = 0; i < 256; ++i) {
      const float normalized = static_cast<float>(i) / 255.0f;
      int q = static_cast<int>(std::lrint(normalized * scale));
      q = std::max(-128, std::min(127, q));
      lut_[static_cast<size_t>(i)] = static_cast<int8_t>(q);
    }
  }

  void quantize(const std::vector<uint8_t>& rgb, int8_t* dst, size_t count) const {
    if (rgb.size() != count) throw std::runtime_error("RGB input size mismatch");
    for (size_t i = 0; i < count; ++i) dst[i] = lut_[rgb[i]];
  }

  std::vector<int8_t> quantize_copy(const std::vector<uint8_t>& rgb) const {
    std::vector<int8_t> out(rgb.size());
    quantize(rgb, out.data(), out.size());
    return out;
  }

  int fix_point() const { return fix_point_; }

 private:
  int fix_point_;
  std::array<int8_t, 256> lut_{};
};

class RunnerContext {
 public:
  explicit RunnerContext(const xir::Subgraph* dpu) {
    runner_ = vart::Runner::create_runner(dpu, "run");
    if (!runner_) throw std::runtime_error("Failed to create VART runner");

    auto inputs = runner_->get_input_tensors();
    auto outputs = runner_->get_output_tensors();
    if (inputs.size() != 1 || outputs.size() != 1) {
      throw std::runtime_error("This benchmark expects exactly one input and one output tensor");
    }
    input_tensor_ = inputs[0];
    output_tensor_ = outputs[0];

    const auto in_shape = input_tensor_->get_shape();
    const auto out_shape = output_tensor_->get_shape();
    if (in_shape.size() != 4 || in_shape[0] != 1 || in_shape[3] != 3) {
      throw std::runtime_error("Expected input shape [1,H,W,3]");
    }
    if (out_shape.empty() || out_shape[0] != 1) {
      throw std::runtime_error("Expected output batch size 1");
    }

    height_ = in_shape[1];
    width_ = in_shape[2];
    channels_ = in_shape[3];
    input_elements_ = static_cast<size_t>(input_tensor_->get_element_num());
    output_elements_ = static_cast<size_t>(output_tensor_->get_element_num());
    input_fix_ = input_tensor_->get_attr<int>("fix_point");
    output_fix_ = output_tensor_->get_attr<int>("fix_point");

    input_.resize(input_elements_);
    output_.resize(output_elements_);
    input_tb_ = std::make_unique<CpuFlatTensorBuffer>(input_.data(), input_tensor_);
    output_tb_ = std::make_unique<CpuFlatTensorBuffer>(output_.data(), output_tensor_);
    input_ptrs_.push_back(input_tb_.get());
    output_ptrs_.push_back(output_tb_.get());
  }

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }
  int input_fix() const { return input_fix_; }
  int output_fix() const { return output_fix_; }
  size_t input_elements() const { return input_elements_; }
  size_t output_elements() const { return output_elements_; }
  const xir::Tensor* input_tensor() const { return input_tensor_; }
  const xir::Tensor* output_tensor() const { return output_tensor_; }

  int8_t* input_data() { return input_.data(); }
  const int8_t* output_data() const { return output_.data(); }

  void load_prequantized(const std::vector<int8_t>& q) {
    if (q.size() != input_.size()) throw std::runtime_error("Prequantized input size mismatch");
    std::memcpy(input_.data(), q.data(), q.size());
  }

  void execute_only() {
    auto job = runner_->execute_async(input_ptrs_, output_ptrs_);
    if (job.second != 0) throw std::runtime_error("execute_async returned non-zero status");
    const int status = runner_->wait(static_cast<int>(job.first), -1);
    if (status != 0) throw std::runtime_error("VART wait returned non-zero status");
  }

  int argmax() const {
    return static_cast<int>(
        std::distance(output_.begin(), std::max_element(output_.begin(), output_.end())));
  }

  int host_to_host(const std::vector<uint8_t>& rgb, const QuantizerLUT& quantizer) {
    quantizer.quantize(rgb, input_.data(), input_.size());
    execute_only();
    return argmax();
  }

 private:
  std::unique_ptr<vart::Runner> runner_;
  const xir::Tensor* input_tensor_ = nullptr;
  const xir::Tensor* output_tensor_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;
  int input_fix_ = 0;
  int output_fix_ = 0;
  size_t input_elements_ = 0;
  size_t output_elements_ = 0;
  std::vector<int8_t> input_;
  std::vector<int8_t> output_;
  std::unique_ptr<CpuFlatTensorBuffer> input_tb_;
  std::unique_ptr<CpuFlatTensorBuffer> output_tb_;
  std::vector<vart::TensorBuffer*> input_ptrs_;
  std::vector<vart::TensorBuffer*> output_ptrs_;
};

static double percentile_sorted(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  if (sorted.size() == 1) return sorted[0];
  const double pos = (p / 100.0) * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  const double frac = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static LatencyStats make_stats(std::vector<double> latencies_ms,
                               uint64_t count, double elapsed_s) {
  LatencyStats s;
  s.count = count;
  s.elapsed_s = elapsed_s;
  s.fps = elapsed_s > 0.0 ? static_cast<double>(count) / elapsed_s : 0.0;
  if (latencies_ms.empty()) return s;

  const double sum = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
  s.mean_ms = sum / latencies_ms.size();
  double sq = 0.0;
  for (double x : latencies_ms) {
    const double d = x - s.mean_ms;
    sq += d * d;
  }
  s.std_ms = std::sqrt(sq / latencies_ms.size());
  std::sort(latencies_ms.begin(), latencies_ms.end());
  s.min_ms = latencies_ms.front();
  s.max_ms = latencies_ms.back();
  s.median_ms = percentile_sorted(latencies_ms, 50.0);
  s.p90_ms = percentile_sorted(latencies_ms, 90.0);
  s.p95_ms = percentile_sorted(latencies_ms, 95.0);
  s.p99_ms = percentile_sorted(latencies_ms, 99.0);
  s.p999_ms = percentile_sorted(latencies_ms, 99.9);
  return s;
}

static double mean_field(const std::vector<LatencyStats>& runs,
                         double LatencyStats::*field) {
  if (runs.empty()) return 0.0;
  double sum = 0.0;
  for (const auto& r : runs) sum += r.*field;
  return sum / static_cast<double>(runs.size());
}

static std::vector<LatencyStats> benchmark_single_model_only(
    RunnerContext& ctx, const std::vector<int8_t>& input, int warmup,
    int runs, double seconds, int power_sample_ms) {
  ctx.load_prequantized(input);
  for (int i = 0; i < warmup; ++i) ctx.execute_only();

  std::vector<LatencyStats> out;
  for (int r = 0; r < runs; ++r) {
    std::vector<double> lat;
    lat.reserve(1000000);
    uint64_t count = 0;
    PowerSampler power(power_sample_ms);
    power.start();
    const auto start_run = Clock::now();
    const auto deadline = start_run + std::chrono::duration_cast<Clock::duration>(
                                        std::chrono::duration<double>(seconds));
    while (Clock::now() < deadline) {
      const auto t0 = Clock::now();
      ctx.execute_only();
      const auto t1 = Clock::now();
      lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      ++count;
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - start_run).count();
    auto stats = make_stats(std::move(lat), count, elapsed);
    stats.power = power.stop();
    out.push_back(std::move(stats));
    std::cout << "  model-only run " << (r + 1) << ": " << std::fixed
              << std::setprecision(2) << out.back().fps << " FPS | "
              << std::setprecision(4) << out.back().mean_ms << " ms\n";
  }
  return out;
}

static std::vector<LatencyStats> benchmark_single_host(
    RunnerContext& ctx, const std::vector<uint8_t>& rgb,
    const QuantizerLUT& quantizer, int warmup, int runs, double seconds,
    int power_sample_ms) {
  for (int i = 0; i < warmup; ++i) (void)ctx.host_to_host(rgb, quantizer);

  std::vector<LatencyStats> out;
  for (int r = 0; r < runs; ++r) {
    std::vector<double> lat;
    lat.reserve(1000000);
    uint64_t count = 0;
    PowerSampler power(power_sample_ms);
    power.start();
    const auto start_run = Clock::now();
    const auto deadline = start_run + std::chrono::duration_cast<Clock::duration>(
                                        std::chrono::duration<double>(seconds));
    while (Clock::now() < deadline) {
      const auto t0 = Clock::now();
      (void)ctx.host_to_host(rgb, quantizer);
      const auto t1 = Clock::now();
      lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      ++count;
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - start_run).count();
    auto stats = make_stats(std::move(lat), count, elapsed);
    stats.power = power.stop();
    out.push_back(std::move(stats));
    std::cout << "  host-to-host run " << (r + 1) << ": " << std::fixed
              << std::setprecision(2) << out.back().fps << " FPS | "
              << std::setprecision(4) << out.back().mean_ms << " ms\n";
  }
  return out;
}

static WorkloadRun run_fixed_memory_single(RunnerContext& ctx,
                                           const std::vector<MemoryImage>& images,
                                           const QuantizerLUT& q,
                                           int power_sample_ms) {
  std::vector<double> lat;
  lat.reserve(images.size());
  uint64_t correct = 0;
  PowerSampler power(power_sample_ms);
  power.start();
  const auto wall0 = Clock::now();
  for (const auto& img : images) {
    const auto t0 = Clock::now();
    int pred = ctx.host_to_host(img.rgb, q);
    const auto t1 = Clock::now();
    lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    if (pred == img.label) ++correct;
  }
  const double elapsed = std::chrono::duration<double>(Clock::now() - wall0).count();
  WorkloadRun w;
  w.images = images.size();
  w.correct = correct;
  w.accuracy = images.empty() ? 0.0 : static_cast<double>(correct) / images.size();
  w.perf = make_stats(std::move(lat), images.size(), elapsed);
  w.perf.power = power.stop();
  return w;
}

static WorkloadRun run_fixed_end_to_end_single(RunnerContext& ctx,
                                                const std::vector<ImageRef>& refs,
                                                const QuantizerLUT& q,
                                                int power_sample_ms) {
  std::vector<double> lat;
  lat.reserve(refs.size());
  uint64_t correct = 0;
  PowerSampler power(power_sample_ms);
  power.start();
  const auto wall0 = Clock::now();
  for (const auto& ref : refs) {
    const auto t0 = Clock::now();
    auto rgb = load_rgb_resized(ref.path, ctx.width(), ctx.height());
    int pred = ctx.host_to_host(rgb, q);
    const auto t1 = Clock::now();
    lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    if (pred == ref.label) ++correct;
  }
  const double elapsed = std::chrono::duration<double>(Clock::now() - wall0).count();
  WorkloadRun w;
  w.images = refs.size();
  w.correct = correct;
  w.accuracy = refs.empty() ? 0.0 : static_cast<double>(correct) / refs.size();
  w.perf = make_stats(std::move(lat), refs.size(), elapsed);
  w.perf.power = power.stop();
  return w;
}

static LatencyStats run_parallel_sustained(const xir::Subgraph* dpu,
                                            const QuantizerLUT& q,
                                            const std::vector<MemoryImage>& images,
                                            const std::vector<int8_t>& prequantized,
                                            int workers, int warmup_per_worker,
                                            double seconds, bool model_only,
                                            int power_sample_ms) {
  std::vector<std::unique_ptr<RunnerContext>> contexts;
  contexts.reserve(workers);
  for (int i = 0; i < workers; ++i) {
    contexts.push_back(std::make_unique<RunnerContext>(dpu));
    if (model_only) contexts.back()->load_prequantized(prequantized);
  }

  for (int w = 0; w < workers; ++w) {
    for (int i = 0; i < warmup_per_worker; ++i) {
      if (model_only) {
        contexts[w]->execute_only();
      } else {
        (void)contexts[w]->host_to_host(images[static_cast<size_t>(i) % images.size()].rgb, q);
      }
    }
  }

  struct Local {
    uint64_t count = 0;
    std::vector<double> lat;
  };
  std::vector<Local> locals(static_cast<size_t>(workers));
  for (auto& l : locals) l.lat.reserve(500000);

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  Clock::time_point deadline;
  std::vector<std::thread> threads;
  threads.reserve(workers);

  for (int w = 0; w < workers; ++w) {
    threads.emplace_back([&, w]() {
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      size_t image_idx = static_cast<size_t>(w) % images.size();
      while (Clock::now() < deadline) {
        const auto t0 = Clock::now();
        if (model_only) {
          contexts[w]->execute_only();
        } else {
          (void)contexts[w]->host_to_host(images[image_idx].rgb, q);
          image_idx = (image_idx + 1) % images.size();
        }
        const auto t1 = Clock::now();
        locals[w].lat.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        ++locals[w].count;
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != workers) std::this_thread::yield();
  PowerSampler power(power_sample_ms);
  power.start();
  const auto wall0 = Clock::now();
  deadline = wall0 + std::chrono::duration_cast<Clock::duration>(
                         std::chrono::duration<double>(seconds));
  go.store(true, std::memory_order_release);
  for (auto& t : threads) t.join();
  const double elapsed = std::chrono::duration<double>(Clock::now() - wall0).count();

  uint64_t total_count = 0;
  size_t total_lat = 0;
  for (const auto& l : locals) {
    total_count += l.count;
    total_lat += l.lat.size();
  }
  std::vector<double> merged;
  merged.reserve(total_lat);
  for (auto& l : locals) {
    merged.insert(merged.end(), l.lat.begin(), l.lat.end());
  }
  auto stats = make_stats(std::move(merged), total_count, elapsed);
  stats.power = power.stop();
  return stats;
}

static std::vector<SweepPoint> run_worker_sweep(const xir::Subgraph* dpu,
                                                 const QuantizerLUT& q,
                                                 const std::vector<MemoryImage>& images,
                                                 const std::vector<int8_t>& prequantized,
                                                 int max_workers,
                                                 int warmup_per_worker,
                                                 int runs, double seconds,
                                                 bool model_only,
                                                 int power_sample_ms) {
  std::vector<SweepPoint> sweep;
  for (int workers = 1; workers <= max_workers; ++workers) {
    SweepPoint p;
    p.workers = workers;
    std::cout << "  workers=" << workers << (model_only ? " model-only" : " host-to-host") << "\n";
    for (int r = 0; r < runs; ++r) {
      auto result = run_parallel_sustained(dpu, q, images, prequantized,
                                           workers, warmup_per_worker,
                                           seconds, model_only, power_sample_ms);
      std::cout << "    run " << (r + 1) << ": " << std::fixed << std::setprecision(2)
                << result.fps << " FPS | mean latency " << std::setprecision(4)
                << result.mean_ms << " ms\n";
      p.runs.push_back(result);
    }
    sweep.push_back(std::move(p));
  }
  return sweep;
}

static int best_workers(const std::vector<SweepPoint>& sweep) {
  double best = -1.0;
  int workers = 1;
  for (const auto& p : sweep) {
    double sum = 0.0;
    for (const auto& r : p.runs) sum += r.fps;
    const double avg = p.runs.empty() ? 0.0 : sum / p.runs.size();
    if (avg > best) {
      best = avg;
      workers = p.workers;
    }
  }
  return workers;
}

static std::vector<LatencyStats> confirm_parallel(const xir::Subgraph* dpu,
                                                   const QuantizerLUT& q,
                                                   const std::vector<MemoryImage>& images,
                                                   const std::vector<int8_t>& prequantized,
                                                   int workers, int warmup,
                                                   int runs, double seconds,
                                                   bool model_only,
                                                   int power_sample_ms) {
  std::vector<LatencyStats> out;
  for (int r = 0; r < runs; ++r) {
    auto x = run_parallel_sustained(dpu, q, images, prequantized, workers,
                                    warmup, seconds, model_only, power_sample_ms);
    std::cout << "  confirmed " << (model_only ? "model" : "host") << " run "
              << (r + 1) << " workers=" << workers << ": " << std::fixed
              << std::setprecision(2) << x.fps << " FPS\n";
    out.push_back(x);
  }
  return out;
}

static WorkloadRun run_fixed_memory_parallel(const xir::Subgraph* dpu,
                                              const QuantizerLUT& q,
                                              const std::vector<MemoryImage>& images,
                                              int workers, int warmup_per_worker,
                                              int power_sample_ms) {
  std::vector<std::unique_ptr<RunnerContext>> contexts;
  contexts.reserve(workers);
  for (int i = 0; i < workers; ++i) contexts.push_back(std::make_unique<RunnerContext>(dpu));
  for (int w = 0; w < workers; ++w) {
    for (int i = 0; i < warmup_per_worker; ++i) {
      (void)contexts[w]->host_to_host(images[static_cast<size_t>(i) % images.size()].rgb, q);
    }
  }

  struct Local {
    uint64_t correct = 0;
    std::vector<double> lat;
  };
  std::vector<Local> locals(static_cast<size_t>(workers));
  std::atomic<size_t> next{0};
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  threads.reserve(workers);

  for (int w = 0; w < workers; ++w) {
    threads.emplace_back([&, w]() {
      locals[w].lat.reserve((images.size() + workers - 1) / workers);
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      while (true) {
        size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= images.size()) break;
        const auto t0 = Clock::now();
        int pred = contexts[w]->host_to_host(images[i].rgb, q);
        const auto t1 = Clock::now();
        locals[w].lat.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (pred == images[i].label) ++locals[w].correct;
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != workers) std::this_thread::yield();
  PowerSampler power(power_sample_ms);
  power.start();
  const auto wall0 = Clock::now();
  go.store(true, std::memory_order_release);
  for (auto& t : threads) t.join();
  const double elapsed = std::chrono::duration<double>(Clock::now() - wall0).count();

  uint64_t correct = 0;
  size_t total_lat = 0;
  for (const auto& l : locals) {
    correct += l.correct;
    total_lat += l.lat.size();
  }
  std::vector<double> merged;
  merged.reserve(total_lat);
  for (auto& l : locals) merged.insert(merged.end(), l.lat.begin(), l.lat.end());

  WorkloadRun w;
  w.images = images.size();
  w.correct = correct;
  w.accuracy = images.empty() ? 0.0 : static_cast<double>(correct) / images.size();
  w.perf = make_stats(std::move(merged), images.size(), elapsed);
  w.perf.power = power.stop();
  return w;
}


static WorkloadRun run_full_end_to_end_parallel(const xir::Subgraph* dpu,
                                                 const QuantizerLUT& q,
                                                 const std::vector<ImageRef>& refs,
                                                 int workers,
                                                 int warmup_per_worker,
                                                 int power_sample_ms) {
  if (refs.empty()) throw std::runtime_error("Full end-to-end dataset is empty");

  std::vector<std::unique_ptr<RunnerContext>> contexts;
  contexts.reserve(static_cast<size_t>(workers));
  for (int i = 0; i < workers; ++i) {
    contexts.push_back(std::make_unique<RunnerContext>(dpu));
  }

  // Warm-up is deliberately outside the timed region. The warm-up image is
  // loaded once so disk/decode work does not contaminate the measured workload.
  auto warm_rgb = load_rgb_resized(refs.front().path,
                                   contexts.front()->width(),
                                   contexts.front()->height());
  for (int w = 0; w < workers; ++w) {
    for (int i = 0; i < warmup_per_worker; ++i) {
      (void)contexts[w]->host_to_host(warm_rgb, q);
    }
  }

  struct Local {
    uint64_t correct = 0;
    std::vector<double> lat;
  };
  std::vector<Local> locals(static_cast<size_t>(workers));
  std::atomic<size_t> next{0};
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers));

  for (int w = 0; w < workers; ++w) {
    threads.emplace_back([&, w]() {
      locals[w].lat.reserve((refs.size() + static_cast<size_t>(workers) - 1) /
                            static_cast<size_t>(workers));
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
      while (true) {
        const size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= refs.size()) break;
        const auto t0 = Clock::now();
        auto rgb = load_rgb_resized(refs[i].path,
                                    contexts[w]->width(),
                                    contexts[w]->height());
        const int pred = contexts[w]->host_to_host(rgb, q);
        const auto t1 = Clock::now();
        locals[w].lat.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (pred == refs[i].label) ++locals[w].correct;
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != workers) std::this_thread::yield();
  PowerSampler power(power_sample_ms);
  power.start();
  const auto wall0 = Clock::now();
  go.store(true, std::memory_order_release);
  for (auto& t : threads) t.join();
  const double elapsed = std::chrono::duration<double>(Clock::now() - wall0).count();

  uint64_t correct = 0;
  size_t total_lat = 0;
  for (const auto& l : locals) {
    correct += l.correct;
    total_lat += l.lat.size();
  }
  std::vector<double> merged;
  merged.reserve(total_lat);
  for (auto& l : locals) merged.insert(merged.end(), l.lat.begin(), l.lat.end());

  WorkloadRun out;
  out.images = refs.size();
  out.correct = correct;
  out.accuracy = static_cast<double>(correct) / static_cast<double>(refs.size());
  out.perf = make_stats(std::move(merged), refs.size(), elapsed);
  out.perf.power = power.stop();
  return out;
}

static std::vector<WorkloadSweepPoint> run_full_e2e_worker_sweep(
    const xir::Subgraph* dpu,
    const QuantizerLUT& q,
    const std::vector<ImageRef>& refs,
    int max_workers,
    int warmup_per_worker,
    int runs,
    int power_sample_ms) {
  std::vector<WorkloadSweepPoint> sweep;
  for (int workers = 1; workers <= max_workers; ++workers) {
    WorkloadSweepPoint point;
    point.workers = workers;
    std::cout << "  workers=" << workers << " full end-to-end\n";
    for (int r = 0; r < runs; ++r) {
      auto result = run_full_end_to_end_parallel(dpu, q, refs, workers,
                                                 warmup_per_worker,
                                                 power_sample_ms);
      std::cout << "    run " << (r + 1) << ": " << std::fixed
                << std::setprecision(6) << result.perf.elapsed_s << " s | "
                << std::setprecision(2) << result.perf.fps << " img/s | acc "
                << result.accuracy * 100.0 << "%";
      if (result.perf.power.available) {
        std::cout << " | power " << result.perf.power.mean_w << " W";
      }
      std::cout << "\n";
      point.runs.push_back(std::move(result));
    }
    sweep.push_back(std::move(point));
  }
  return sweep;
}

static int best_workload_workers(const std::vector<WorkloadSweepPoint>& sweep) {
  double best_fps = -1.0;
  int best = 1;
  for (const auto& p : sweep) {
    double sum = 0.0;
    for (const auto& r : p.runs) sum += r.perf.fps;
    const double mean = p.runs.empty() ? 0.0 : sum / static_cast<double>(p.runs.size());
    if (mean > best_fps) {
      best_fps = mean;
      best = p.workers;
    }
  }
  return best;
}

static std::string self_exe_path() {
  std::array<char, 4096> buf{};
  ssize_t n = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1);
  if (n <= 0) throw std::runtime_error("Cannot resolve /proc/self/exe");
  buf[static_cast<size_t>(n)] = '\0';
  return std::string(buf.data());
}

static double run_one_cold_start(const std::string& exe, const Options& o,
                                 const fs::path& image) {
  const auto t0 = Clock::now();
  pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("fork() failed");
  if (pid == 0) {
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, STDOUT_FILENO);
      dup2(devnull, STDERR_FILENO);
      close(devnull);
    }
    execl(exe.c_str(), exe.c_str(), "--cold-child", "--model", o.model.c_str(),
          "--cold-image", image.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) throw std::runtime_error("waitpid() failed");
  const auto t1 = Clock::now();
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("Cold-start child failed");
  }
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int cold_child_main(const Options& o) {
  if (o.cold_image.empty()) throw std::runtime_error("--cold-image required in child mode");
  auto graph = xir::Graph::deserialize(o.model);
  auto* dpu = find_single_dpu_subgraph(graph.get());
  RunnerContext ctx(dpu);
  QuantizerLUT q(ctx.input_fix());
  auto rgb = load_rgb_resized(o.cold_image, ctx.width(), ctx.height());
  (void)ctx.host_to_host(rgb, q);
  return 0;
}

static std::string now_timestamp() {
  auto now = std::chrono::system_clock::now();
  std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&tt, &tm);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return oss.str();
}

static std::string command_output(const std::string& cmd) {
  std::array<char, 512> buffer{};
  std::string result;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return "";
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) result += buffer.data();
  pclose(pipe);
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
  return result;
}

static std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (unsigned char c : s) {
    switch (c) {
      case '"': o << "\\\""; break;
      case '\\': o << "\\\\"; break;
      case '\b': o << "\\b"; break;
      case '\f': o << "\\f"; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default:
        if (c < 0x20) {
          o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
            << std::dec << std::setfill(' ');
        } else {
          o << static_cast<char>(c);
        }
    }
  }
  return o.str();
}

static void write_string_array(std::ostream& os, const std::vector<std::string>& v,
                               int indent) {
  os << "[\n";
  for (size_t i = 0; i < v.size(); ++i) {
    os << std::string(indent + 2, ' ') << "\"" << json_escape(v[i]) << "\"";
    if (i + 1 != v.size()) os << ",";
    os << "\n";
  }
  os << std::string(indent, ' ') << "]";
}


static PowerStats aggregate_power_latency(const std::vector<LatencyStats>& runs) {
  PowerStats out;
  std::vector<double> means;
  double energy_sum = 0.0;
  double duration_sum = 0.0;
  uint64_t samples_sum = 0;
  double min_obs = std::numeric_limits<double>::infinity();
  double max_obs = -std::numeric_limits<double>::infinity();
  for (const auto& r : runs) {
    if (!r.power.available) continue;
    if (out.sensor_name.empty()) {
      out.sensor_name = r.power.sensor_name;
      out.sensor_path = r.power.sensor_path;
    }
    means.push_back(r.power.mean_w);
    energy_sum += r.power.energy_j;
    duration_sum += r.power.duration_s;
    samples_sum += r.power.samples;
    min_obs = std::min(min_obs, r.power.min_w);
    max_obs = std::max(max_obs, r.power.max_w);
  }
  if (means.empty()) return out;
  out.available = true;
  out.samples = samples_sum;
  out.duration_s = duration_sum / static_cast<double>(means.size());
  out.energy_j = energy_sum / static_cast<double>(means.size());
  out.min_w = min_obs;
  out.max_w = max_obs;
  out.mean_w = std::accumulate(means.begin(), means.end(), 0.0) /
               static_cast<double>(means.size());
  if (means.size() > 1) {
    double sq = 0.0;
    for (double x : means) {
      const double d = x - out.mean_w;
      sq += d * d;
    }
    out.std_w = std::sqrt(sq / static_cast<double>(means.size() - 1));
    const double half = 1.96 * out.std_w / std::sqrt(static_cast<double>(means.size()));
    out.ci95_low_w = out.mean_w - half;
    out.ci95_high_w = out.mean_w + half;
  } else {
    out.std_w = runs.front().power.std_w;
    out.ci95_low_w = runs.front().power.ci95_low_w;
    out.ci95_high_w = runs.front().power.ci95_high_w;
  }
  return out;
}

static PowerStats aggregate_power_workloads(const std::vector<WorkloadRun>& runs) {
  std::vector<LatencyStats> tmp;
  tmp.reserve(runs.size());
  for (const auto& r : runs) tmp.push_back(r.perf);
  return aggregate_power_latency(tmp);
}

static void write_power(std::ostream& os, const PowerStats& pwr, int indent) {
  const std::string p(indent, ' ');
  os << "{\n"
     << p << "  \"available\": " << (pwr.available ? "true" : "false");
  if (pwr.available) {
    os << ",\n"
       << p << "  \"sensor_name\": \"" << json_escape(pwr.sensor_name) << "\",\n"
       << p << "  \"sensor_path\": \"" << json_escape(pwr.sensor_path) << "\",\n"
       << p << "  \"samples\": " << pwr.samples << ",\n"
       << p << "  \"measurement_duration_s\": " << pwr.duration_s << ",\n"
       << p << "  \"mean_w\": " << pwr.mean_w << ",\n"
       << p << "  \"min_w\": " << pwr.min_w << ",\n"
       << p << "  \"max_w\": " << pwr.max_w << ",\n"
       << p << "  \"std_w\": " << pwr.std_w << ",\n"
       << p << "  \"ci95_low_w\": " << pwr.ci95_low_w << ",\n"
       << p << "  \"ci95_high_w\": " << pwr.ci95_high_w << ",\n"
       << p << "  \"energy_j\": " << pwr.energy_j << "\n";
  } else {
    os << "\n";
  }
  os << p << "}";
}

static void write_latency(std::ostream& os, const LatencyStats& s, int indent) {
  const std::string p(indent, ' ');
  os << "{\n"
     << p << "  \"inferences\": " << s.count << ",\n"
     << p << "  \"elapsed_s\": " << std::setprecision(12) << s.elapsed_s << ",\n"
     << p << "  \"throughput_images_per_s\": " << s.fps << ",\n"
     << p << "  \"latency_mean_ms\": " << s.mean_ms << ",\n"
     << p << "  \"latency_median_ms\": " << s.median_ms << ",\n"
     << p << "  \"latency_std_ms\": " << s.std_ms << ",\n"
     << p << "  \"latency_min_ms\": " << s.min_ms << ",\n"
     << p << "  \"latency_max_ms\": " << s.max_ms << ",\n"
     << p << "  \"p90_ms\": " << s.p90_ms << ",\n"
     << p << "  \"p95_ms\": " << s.p95_ms << ",\n"
     << p << "  \"p99_ms\": " << s.p99_ms << ",\n"
     << p << "  \"p999_ms\": " << s.p999_ms << ",\n"
     << p << "  \"power\": ";
  write_power(os, s.power, indent + 2);
  os << "\n" << p << "}";
}

static void write_runs(std::ostream& os, const std::vector<LatencyStats>& runs, int indent) {
  os << "[\n";
  for (size_t i = 0; i < runs.size(); ++i) {
    os << std::string(indent + 2, ' ');
    write_latency(os, runs[i], indent + 2);
    if (i + 1 != runs.size()) os << ",";
    os << "\n";
  }
  os << std::string(indent, ' ') << "]";
}

static void write_latency_summary(std::ostream& os,
                                  const std::vector<LatencyStats>& runs,
                                  int indent) {
  const std::string p(indent, ' ');
  os << "{\n"
     << p << "  \"fps_mean_across_runs\": " << mean_field(runs, &LatencyStats::fps) << ",\n"
     << p << "  \"latency_mean_ms_across_runs\": " << mean_field(runs, &LatencyStats::mean_ms) << ",\n"
     << p << "  \"median_ms_across_runs\": " << mean_field(runs, &LatencyStats::median_ms) << ",\n"
     << p << "  \"p90_ms_across_runs\": " << mean_field(runs, &LatencyStats::p90_ms) << ",\n"
     << p << "  \"p95_ms_across_runs\": " << mean_field(runs, &LatencyStats::p95_ms) << ",\n"
     << p << "  \"p99_ms_across_runs\": " << mean_field(runs, &LatencyStats::p99_ms) << ",\n"
     << p << "  \"p999_ms_across_runs\": " << mean_field(runs, &LatencyStats::p999_ms) << ",\n"
     << p << "  \"power_summary\": ";
  write_power(os, aggregate_power_latency(runs), indent + 2);
  os << "\n" << p << "}";
}

static void write_workload_run(std::ostream& os, const WorkloadRun& w, int indent) {
  const std::string p(indent, ' ');
  os << "{\n"
     << p << "  \"images\": " << w.images << ",\n"
     << p << "  \"correct\": " << w.correct << ",\n"
     << p << "  \"accuracy\": " << w.accuracy << ",\n"
     << p << "  \"performance\": ";
  write_latency(os, w.perf, indent + 2);
  os << "\n" << p << "}";
}

static void write_workload_runs(std::ostream& os, const std::vector<WorkloadRun>& runs,
                                int indent) {
  os << "[\n";
  for (size_t i = 0; i < runs.size(); ++i) {
    os << std::string(indent + 2, ' ');
    write_workload_run(os, runs[i], indent + 2);
    if (i + 1 != runs.size()) os << ",";
    os << "\n";
  }
  os << std::string(indent, ' ') << "]";
}

static void write_section_meta(std::ostream& os, const std::string& description,
                               const std::string& purpose,
                               const std::vector<std::string>& includes,
                               const std::vector<std::string>& excludes,
                               const std::vector<std::string>& caveats,
                               int indent) {
  const std::string p(indent, ' ');
  os << p << "\"description\": \"" << json_escape(description) << "\",\n"
     << p << "\"purpose\": \"" << json_escape(purpose) << "\",\n"
     << p << "\"timed_region_includes\": ";
  write_string_array(os, includes, indent);
  os << ",\n" << p << "\"timed_region_excludes\": ";
  write_string_array(os, excludes, indent);
  os << ",\n" << p << "\"caveats\": ";
  write_string_array(os, caveats, indent);
  os << ",\n";
}

static void write_json(const fs::path& path, const Options& o,
                       const RunnerContext& probe,
                       size_t dataset_count, size_t workload_count,
                       const BenchmarkResults& r) {
  std::ofstream os(path);
  if (!os) throw std::runtime_error("Cannot create JSON: " + path.string());
  os << std::setprecision(12);

  const fs::path power_sensor = detect_power_sensor_path();
  const std::string power_sensor_name = power_sensor.empty() ? "" : read_first_line(power_sensor.parent_path() / "name");

  os << "{\n"
     << "  \"schema_version\": 2,\n"
     << "  \"benchmark\": \"TrafficSignNet ZCU104 VART C++ comprehensive suite\",\n"
     << "  \"metadata\": {\n"
     << "    \"model_path\": \"" << json_escape(fs::absolute(o.model).string()) << "\",\n"
     << "    \"dataset_path\": \"" << json_escape(fs::absolute(o.dataset).string()) << "\",\n"
     << "    \"dataset_images_available\": " << dataset_count << ",\n"
     << "    \"fixed_workload_images\": " << workload_count << ",\n"
     << "    \"seed\": " << o.seed << ",\n"
     << "    \"input_tensor_name\": \"" << json_escape(probe.input_tensor()->get_name()) << "\",\n"
     << "    \"input_shape\": [1," << probe.height() << "," << probe.width() << "," << probe.channels() << "],\n"
     << "    \"input_fix_point\": " << probe.input_fix() << ",\n"
     << "    \"output_tensor_name\": \"" << json_escape(probe.output_tensor()->get_name()) << "\",\n"
     << "    \"output_elements\": " << probe.output_elements() << ",\n"
     << "    \"output_fix_point\": " << probe.output_fix() << ",\n"
     << "    \"opencv_version\": \"" << json_escape(CV_VERSION) << "\",\n"
     << "    \"opencv_threads\": " << cv::getNumThreads() << ",\n"
     << "    \"hardware_concurrency\": " << std::thread::hardware_concurrency() << ",\n"
     << "    \"uname\": \"" << json_escape(command_output("uname -a")) << "\",\n"
     << "    \"xdputil_version\": \"" << json_escape(command_output("xdputil --version 2>&1")) << "\",\n"
     << "    \"power_sensor_name\": \"" << json_escape(power_sensor_name) << "\",\n"
     << "    \"power_sensor_path\": \"" << json_escape(power_sensor.string()) << "\",\n"
     << "    \"power_sensor_scope\": \"Linux hwmon power channel exposed by the board image; not automatically claimed as wall-plug or FPGA-fabric-only power\",\n"
     << "    \"official_gtsrb_test_expected_images\": 12630,\n"
     << "    \"official_gtsrb_test_complete\": " << (dataset_count == 12630 ? "true" : "false") << ",\n"
     << "    \"preprocess\": \"OpenCV BGR->RGB, INTER_AREA resize to model input, uint8->INT8 LUT implementing round((pixel/255)*2^fix_point)\",\n"
     << "    \"batch_size\": 1\n"
     << "  },\n"
     << "  \"configuration\": {\n"
     << "    \"warmup\": " << o.warmup << ",\n"
     << "    \"runs\": " << o.runs << ",\n"
     << "    \"seconds_per_run\": " << o.seconds << ",\n"
     << "    \"workload_runs\": " << o.workload_runs << ",\n"
     << "    \"max_workers_tested\": " << o.max_workers << ",\n"
     << "    \"sweep_runs\": " << o.sweep_runs << ",\n"
     << "    \"sweep_seconds\": " << o.sweep_seconds << ",\n"
     << "    \"sweep_warmup_per_worker\": " << o.sweep_warmup << ",\n"
     << "    \"full_e2e_runs_per_worker\": " << o.full_e2e_runs << ",\n"
     << "    \"power_sample_interval_ms\": " << o.power_sample_ms << ",\n"
     << "    \"cold_runs\": " << o.cold_runs << "\n"
     << "  },\n"
     << "  \"benchmarks\": {\n";

  // single model-only
  os << "    \"single_stream_model_only\": {\n";
  write_section_meta(os,
      "Synchronous batch-1 VART/DPU execution with an already quantized INT8 input resident in the runner input buffer.",
      "Primary latency benchmark and fair model-runtime reference for batch-1 comparisons.",
      {"VART execute_async", "blocking wait for completion"},
      {"disk I/O", "image decode", "resize", "RGB conversion", "normalization/quantization", "argmax", "model/runner creation"},
      {"One request is in flight at a time; this is latency-oriented, not maximum two-DPU-core throughput."}, 6);
  os << "      \"summary\": "; write_latency_summary(os, r.single_model_only, 6); os << ",\n"
     << "      \"runs\": "; write_runs(os, r.single_model_only, 6); os << "\n    },\n";

  // single host
  os << "    \"single_stream_host_to_host\": {\n";
  write_section_meta(os,
      "Synchronous batch-1 application-compute path starting from a decoded/resized RGB uint8 image in host RAM.",
      "Measures host preprocessing overhead plus one DPU inference and class decision.",
      {"INT8 LUT quantization", "copy into runner input buffer", "VART execute_async", "blocking wait", "INT8 argmax"},
      {"disk I/O", "PNG/JPEG decode", "resize", "model/runner creation"},
      {"Uses an optimized 256-entry LUT that is mathematically equivalent to round((pixel/255)*2^input_fix_point)."}, 6);
  os << "      \"summary\": "; write_latency_summary(os, r.single_host_to_host, 6); os << ",\n"
     << "      \"runs\": "; write_runs(os, r.single_host_to_host, 6); os << "\n    },\n";

  // fixed memory single
  os << "    \"fixed_workload_memory_single_stream\": {\n";
  write_section_meta(os,
      "Closed workload of different real dataset images already decoded/resized and resident in RAM, processed serially at batch 1.",
      "Answers: if N real images are already in memory, how long until the Nth result is complete on a single synchronous stream?",
      {"per-image INT8 LUT quantization", "DPU inference", "wait", "argmax", "accuracy bookkeeping"},
      {"disk I/O", "decode", "resize", "model/runner creation"},
      {"Total elapsed time is measured directly from the first image start to completion of the last image; FPS is images/that wall time."}, 6);
  os << "      \"runs\": "; write_workload_runs(os, r.fixed_memory_single, 6); os << "\n    },\n";

  // fixed e2e
  os << "    \"fixed_workload_end_to_end_single_stream\": {\n";
  write_section_meta(os,
      "Closed workload of different real image files processed serially from filesystem path to predicted class.",
      "Closest local-file measurement to 'I gave the application N image files; when was the last result ready?'.",
      {"file read", "image decode", "BGR->RGB", "resize", "INT8 LUT quantization", "DPU inference", "wait", "argmax"},
      {"model/runner creation"},
      {"OS page cache is not flushed between repetitions, so later runs can benefit from cached file data."}, 6);
  os << "      \"runs\": "; write_workload_runs(os, r.fixed_end_to_end_single, 6); os << "\n    },\n";

  // full test
  os << "    \"full_dataset_end_to_end\": {\n";
  write_section_meta(os,
      "Serial end-to-end pass over every labeled image file currently present under the dataset directory.",
      "Measures accuracy plus complete file-to-prediction throughput on the available board dataset.",
      {"file read", "decode", "BGR->RGB", "resize", "INT8 LUT quantization", "DPU inference", "wait", "argmax"},
      {"model/runner creation"},
      {"The repository deployment subset contains fewer images than the official 12,630-image GTSRB test set unless you replace/copy the full test directory to the board."}, 6);
  os << "      \"executed\": " << (r.full_test_ran ? "true" : "false");
  if (r.full_test_ran) {
    os << ",\n      \"result\": "; write_workload_run(os, r.full_test_end_to_end, 6); os << "\n";
  } else {
    os << "\n";
  }
  os << "    },\n";

  auto write_sweep = [&](const char* name, const std::vector<SweepPoint>& sweep,
                         bool model_only, int best, bool trailing_comma) {
    os << "    \"" << name << "\": {\n";
    write_section_meta(os,
        model_only ?
          "Concurrent VART runner/worker sweep using 1..N independent Runner instances and prequantized batch-1 input." :
          "Concurrent host-to-host sweep using 1..N independent Runner instances; every inference performs LUT quantization and argmax.",
        model_only ?
          "Finds the worker count that best keeps multiple DPU cores occupied for maximum DPU/VART throughput." :
          "Finds the worker count that maximizes application-compute throughput while including host preprocessing.",
        model_only ? std::vector<std::string>{"independent runners", "concurrent execute/wait", "DPU scheduling"} :
                     std::vector<std::string>{"independent runners", "LUT quantization", "concurrent execute/wait", "argmax"},
        model_only ? std::vector<std::string>{"disk I/O", "decode", "resize", "per-inference quantization", "argmax"} :
                     std::vector<std::string>{"disk I/O", "decode", "resize"},
        {"Each runner has private input/output buffers; runner creation and warmup are outside the timed region.",
         "More workers than physical DPU cores can still be useful to hide scheduling/host gaps, but can also reduce performance."}, 6);
    os << "      \"best_workers_by_mean_fps\": " << best << ",\n"
       << "      \"points\": [\n";
    for (size_t i = 0; i < sweep.size(); ++i) {
      os << "        {\n"
         << "          \"workers\": " << sweep[i].workers << ",\n"
         << "          \"summary\": "; write_latency_summary(os, sweep[i].runs, 10); os << ",\n"
         << "          \"runs\": "; write_runs(os, sweep[i].runs, 10); os << "\n"
         << "        }" << (i + 1 == sweep.size() ? "" : ",") << "\n";
    }
    os << "      ]\n"
       << "    }" << (trailing_comma ? "," : "") << "\n";
  };

  write_sweep("worker_sweep_model_only", r.sweep_model_only, true, r.best_model_workers, true);
  write_sweep("worker_sweep_host_to_host", r.sweep_host_to_host, false, r.best_host_workers, true);

  os << "    \"max_throughput_model_only\": {\n";
  write_section_meta(os,
      "Long sustained confirmation using the worker count selected by the model-only sweep.",
      "Reports the maximum sustained VART/DPU throughput configuration discovered by the suite.",
      {"multiple independent VART runners", "concurrent DPU inference", "blocking wait inside each worker"},
      {"disk I/O", "decode", "resize", "per-inference host preprocessing", "argmax"},
      {"This is a throughput-capacity result and must not replace the single-stream batch-1 latency result."}, 6);
  os << "      \"workers\": " << r.best_model_workers << ",\n"
     << "      \"executed\": " << (!r.max_model_only.empty() ? "true" : "false");
  if (!r.max_model_only.empty()) {
    os << ",\n      \"summary\": "; write_latency_summary(os, r.max_model_only, 6); os << ",\n"
       << "      \"runs\": "; write_runs(os, r.max_model_only, 6); os << "\n";
  } else os << "\n";
  os << "    },\n";

  os << "    \"max_throughput_host_to_host\": {\n";
  write_section_meta(os,
      "Long sustained confirmation using the worker count selected by the host-to-host sweep.",
      "Reports maximum sustained application-compute throughput with optimized INT8 preprocessing.",
      {"multiple independent runners", "LUT quantization", "concurrent DPU inference", "wait", "argmax"},
      {"disk I/O", "decode", "resize"},
      {"This is a maximum-throughput result, not single-request latency."}, 6);
  os << "      \"workers\": " << r.best_host_workers << ",\n"
     << "      \"executed\": " << (!r.max_host_to_host.empty() ? "true" : "false");
  if (!r.max_host_to_host.empty()) {
    os << ",\n      \"summary\": "; write_latency_summary(os, r.max_host_to_host, 6); os << ",\n"
       << "      \"runs\": "; write_runs(os, r.max_host_to_host, 6); os << "\n";
  } else os << "\n";
  os << "    },\n";

  os << "    \"fixed_workload_memory_parallel\": {\n";
  write_section_meta(os,
      "Closed in-memory workload processed concurrently by the best host-to-host worker count discovered in the sweep.",
      "Answers: with both DPU cores/available runtime concurrency exploited, how long until all N in-memory real images are classified?",
      {"parallel worker dispatch", "LUT quantization", "DPU inference", "wait", "argmax"},
      {"disk I/O", "decode", "resize", "runner creation/warmup"},
      {"Total completion wall time is the key metric; individual per-request latencies can increase while total workload completion time decreases."}, 6);
  os << "      \"workers\": " << r.best_host_workers << ",\n"
     << "      \"runs\": "; write_workload_runs(os, r.fixed_memory_parallel, 6); os << "\n    },\n";

  os << "    \"full_dataset_end_to_end_worker_sweep\": {\n";
  write_section_meta(os,
      "Complete file-to-classification pass over every labeled test image using 1..N independent VART runners/threads.",
      "Primary real-workload capacity benchmark: answers how long a complete image set takes when host I/O, decode, resize, quantization and all available DPU concurrency are used.",
      {"file read", "image decode", "BGR->RGB", "resize", "INT8 LUT quantization", "parallel worker dispatch", "VART/DPU inference", "wait", "argmax", "accuracy bookkeeping"},
      {"initial XMODEL deserialize", "runner creation", "warmup"},
      {"The official GTSRB test split contains 12,630 images across 43 classes; if fewer files are present, the report marks the dataset incomplete.",
       "OS page cache is not forcibly flushed, so repeated runs may benefit from caching.",
       "Use this section for real batch-of-files completion time and maximum end-to-end throughput; do not use it as single-request latency."}, 6);
  os << "      \"expected_official_images\": 12630,\n"
     << "      \"dataset_complete\": " << (dataset_count == 12630 ? "true" : "false") << ",\n"
     << "      \"best_workers_by_mean_fps\": " << r.best_full_e2e_workers << ",\n"
     << "      \"points\": [\n";
  for (size_t i = 0; i < r.full_e2e_worker_sweep.size(); ++i) {
    const auto& point = r.full_e2e_worker_sweep[i];
    os << "        {\n"
       << "          \"workers\": " << point.workers << ",\n"
       << "          \"power_summary\": ";
    write_power(os, aggregate_power_workloads(point.runs), 10);
    os << ",\n          \"runs\": ";
    write_workload_runs(os, point.runs, 10);
    os << "\n        }" << (i + 1 == r.full_e2e_worker_sweep.size() ? "" : ",") << "\n";
  }
  os << "      ]\n"
     << "    },\n";

  os << "    \"cold_start\": {\n";
  write_section_meta(os,
      "Fresh Linux process startup through completion of the first classification.",
      "Measures startup cost separately from steady-state inference.",
      {"fork/exec of fresh benchmark process", "dynamic loader/process startup", "XIR graph deserialize", "VART runner creation", "image read/decode/resize", "quantization", "first DPU inference", "argmax", "process exit"},
      {"parent benchmark startup"},
      {"The parent measures child process wall-clock duration; this intentionally includes more than inference."}, 6);
  os << "      \"executed\": " << (!r.cold_start_ms.empty() ? "true" : "false");
  if (!r.cold_start_ms.empty()) {
    auto cs = make_stats(r.cold_start_ms, r.cold_start_ms.size(),
                         std::accumulate(r.cold_start_ms.begin(), r.cold_start_ms.end(), 0.0) / 1000.0);
    os << ",\n      \"runs_ms\": [";
    for (size_t i = 0; i < r.cold_start_ms.size(); ++i) {
      if (i) os << ", ";
      os << r.cold_start_ms[i];
    }
    os << "],\n      \"mean_ms\": " << cs.mean_ms
       << ",\n      \"median_ms\": " << cs.median_ms
       << ",\n      \"min_ms\": " << cs.min_ms
       << ",\n      \"max_ms\": " << cs.max_ms << "\n";
  } else os << "\n";
  os << "    }\n";

  os << "  }\n"
     << "}\n";
}

static double avg_workload_elapsed(const std::vector<WorkloadRun>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (const auto& x : v) s += x.perf.elapsed_s;
  return s / v.size();
}

static double avg_workload_fps(const std::vector<WorkloadRun>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (const auto& x : v) s += x.perf.fps;
  return s / v.size();
}

static double avg_workload_acc(const std::vector<WorkloadRun>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (const auto& x : v) s += x.accuracy;
  return s / v.size();
}

static void write_power_md(std::ostream& os, const PowerStats& p,
                           uint64_t images_per_run = 0) {
  os << "**Power / energy**  \n";
  if (!p.available) {
    os << "Power sensor unavailable for this measurement.\n\n";
    return;
  }
  os << "- Sensor: `" << p.sensor_name << "` (`" << p.sensor_path << "`)\n"
     << "- Mean power: **" << std::setprecision(3) << p.mean_w << " W**\n"
     << "- Minimum observed: **" << p.min_w << " W**\n"
     << "- Maximum observed: **" << p.max_w << " W**\n"
     << "- 95% CI of mean power: **[" << p.ci95_low_w << ", "
     << p.ci95_high_w << "] W**\n"
     << "- Power samples: **" << p.samples << "**\n"
     << "- Mean energy per measured run: **" << p.energy_j << " J**\n";
  if (images_per_run > 0) {
    os << "- Approx. energy/image: **" << (p.energy_j / static_cast<double>(images_per_run))
       << " J/image**\n";
  }
  os << "\n";
}

static double point_mean_elapsed(const WorkloadSweepPoint& p) {
  return avg_workload_elapsed(p.runs);
}

static double point_mean_fps(const WorkloadSweepPoint& p) {
  return avg_workload_fps(p.runs);
}

static double point_mean_acc(const WorkloadSweepPoint& p) {
  return avg_workload_acc(p.runs);
}

static void write_markdown(const fs::path& path, const Options& o,
                           size_t dataset_count, size_t workload_count,
                           const BenchmarkResults& r) {
  std::ofstream os(path);
  if (!os) throw std::runtime_error("Cannot create Markdown: " + path.string());
  os << std::fixed;
  const fs::path power_path = detect_power_sensor_path();
  const std::string power_name = power_path.empty() ? "unavailable" :
      read_first_line(power_path.parent_path() / "name");

  os << "# TrafficSignNet ZCU104 C++ benchmark report\n\n"
     << "Model: `" << o.model << "`  \n"
     << "Dataset: `" << o.dataset << "` (**" << dataset_count << "** labeled images found)  \n"
     << "Official GTSRB test size expected: **12,630 images / 43 classes**  \n"
     << "Official test complete on board: **" << (dataset_count == 12630 ? "YES" : "NO") << "**  \n"
     << "Fixed workload: **" << workload_count << "** different images  \n"
     << "Batch size per request: **1**  \n"
     << "Power sensor: `" << power_name << "` (`" << power_path.string() << "`)  \n"
     << "Power sampling interval: **" << o.power_sample_ms << " ms**\n\n"
     << "> Power terminology: the report records the Linux `hwmon` channel exposed by the board image. "
        "It is not automatically labeled as wall-plug power or FPGA-fabric-only power. The 95% confidence interval "
        "is computed from run means when repeated runs exist; for a single run it is an approximate sample-based interval.\n\n";

  os << "## 1. Single-stream model-only\n\n"
     << "**How it works:** one batch-1 request at a time. The INT8 input is already quantized and already resident in the runner input buffer. "
        "The timed region is essentially `execute_async -> DPU -> wait`.\n\n"
     << "**Includes:** VART submission and blocking DPU completion.  \n"
     << "**Excludes:** file I/O, decode, resize, color conversion, quantization, argmax, model/runner creation.  \n"
     << "**Use to compare:** pure batch-1 accelerator/runtime latency across DPU, CPU/GPU model-only paths and HLS4ML when their timed regions are equivalent.  \n"
     << "**Do not use for:** real file-processing throughput or maximum multi-core DPU capacity.\n\n"
     << "- Mean FPS: **" << std::setprecision(2) << mean_field(r.single_model_only, &LatencyStats::fps) << "**\n"
     << "- Mean latency: **" << std::setprecision(4) << mean_field(r.single_model_only, &LatencyStats::mean_ms) << " ms**\n"
     << "- Mean P99: **" << mean_field(r.single_model_only, &LatencyStats::p99_ms) << " ms**\n\n";
  write_power_md(os, aggregate_power_latency(r.single_model_only));

  os << "## 2. Single-stream host-to-host\n\n"
     << "**How it works:** starts from a decoded/resized RGB `uint8` image already in RAM, then performs LUT INT8 quantization, DPU execution, wait and argmax.\n\n"
     << "**Includes:** host quantization/copy, VART/DPU inference, synchronization and class decision.  \n"
     << "**Excludes:** disk I/O, image decode, resize and model/runner creation.  \n"
     << "**Use to compare:** application-compute overhead around batch-1 inference.  \n"
     << "**Do not use for:** complete file-to-result latency.\n\n"
     << "- Mean FPS: **" << std::setprecision(2) << mean_field(r.single_host_to_host, &LatencyStats::fps) << "**\n"
     << "- Mean latency: **" << std::setprecision(4) << mean_field(r.single_host_to_host, &LatencyStats::mean_ms) << " ms**\n\n";
  write_power_md(os, aggregate_power_latency(r.single_host_to_host));

  os << "## 3. Fixed workload in memory - single stream\n\n"
     << "**How it works:** N different real images are decoded/resized before timing and remain in RAM. They are then classified serially, one after another.\n\n"
     << "**Use to compare:** exact completion time for an already-in-memory workload.  \n"
     << "**Do not use for:** storage/decode performance or maximum parallel throughput.\n\n"
     << "- Images/run: **" << workload_count << "**\n"
     << "- Mean total completion time: **" << std::setprecision(6) << avg_workload_elapsed(r.fixed_memory_single) << " s**\n"
     << "- Mean throughput: **" << std::setprecision(2) << avg_workload_fps(r.fixed_memory_single) << " img/s**\n"
     << "- Mean accuracy: **" << avg_workload_acc(r.fixed_memory_single) * 100.0 << "%**\n\n";
  write_power_md(os, aggregate_power_workloads(r.fixed_memory_single), workload_count);

  os << "## 4. Fixed workload end-to-end - single stream\n\n"
     << "**How it works:** N image files are processed serially from filesystem path to final predicted class.\n\n"
     << "**Includes:** file read, decode, BGR->RGB, resize, INT8 quantization, DPU, wait and argmax.  \n"
     << "**Excludes:** initial XMODEL/runner creation.  \n"
     << "**Use to compare:** realistic local-file processing when only one request is active at a time.  \n"
     << "**Do not use for:** pure DPU latency. OS page cache is not flushed between repetitions.\n\n"
     << "- Images/run: **" << workload_count << "**\n"
     << "- Mean total completion time: **" << std::setprecision(6) << avg_workload_elapsed(r.fixed_end_to_end_single) << " s**\n"
     << "- Mean throughput: **" << std::setprecision(2) << avg_workload_fps(r.fixed_end_to_end_single) << " img/s**\n"
     << "- Mean accuracy: **" << avg_workload_acc(r.fixed_end_to_end_single) * 100.0 << "%**\n\n";
  write_power_md(os, aggregate_power_workloads(r.fixed_end_to_end_single), workload_count);

  os << "## 5. Full available dataset end-to-end - single stream\n\n"
     << "**How it works:** exactly the same file-to-result path as section 4, but over every labeled image currently present under the dataset directory.\n\n"
     << "**Use to compare:** full-dataset accuracy and single-stream real completion time.  \n"
     << "**Do not use for:** maximum platform throughput; section 10 performs the same real workload with 1-4 workers.\n\n";
  if (r.full_test_ran) {
    os << "- Images: **" << r.full_test_end_to_end.images << "**\n"
       << "- Total time: **" << std::setprecision(6) << r.full_test_end_to_end.perf.elapsed_s << " s**\n"
       << "- Throughput: **" << std::setprecision(2) << r.full_test_end_to_end.perf.fps << " img/s**\n"
       << "- Accuracy: **" << r.full_test_end_to_end.accuracy * 100.0 << "%**\n\n";
    write_power_md(os, r.full_test_end_to_end.perf.power, r.full_test_end_to_end.images);
  } else {
    os << "Skipped.\n\n";
  }

  auto sweep_table = [&](const char* title, const std::vector<SweepPoint>& sweep,
                         int best, const char* use_for, const char* not_for) {
    os << "## " << title << "\n\n"
       << "**How it works:** independent VART runners/threads execute concurrently with private input/output buffers; runner creation and warmup are outside timing.\n\n"
       << "**Use to compare:** " << use_for << "  \n"
       << "**Do not use for:** " << not_for << "\n\n"
       << "| Workers | Mean FPS | Mean latency/request (ms) | Mean power (W) | Min W | Max W | 95% CI mean W |\n"
       << "|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& point : sweep) {
      const auto power = aggregate_power_latency(point.runs);
      os << "| " << point.workers << " | " << std::setprecision(2)
         << mean_field(point.runs, &LatencyStats::fps) << " | " << std::setprecision(4)
         << mean_field(point.runs, &LatencyStats::mean_ms) << " | ";
      if (power.available) {
        os << std::setprecision(3) << power.mean_w << " | " << power.min_w << " | "
           << power.max_w << " | [" << power.ci95_low_w << ", " << power.ci95_high_w << "] |\n";
      } else {
        os << "N/A | N/A | N/A | N/A |\n";
      }
    }
    os << "\nSelected best worker count by mean FPS: **" << best << "**\n\n";
  };

  sweep_table("6. Worker sweep - model-only maximum throughput", r.sweep_model_only,
              r.best_model_workers,
              "maximum VART/DPU capacity with already-quantized data.",
              "single-request latency or file-to-result performance.");
  sweep_table("7. Worker sweep - host-to-host maximum throughput", r.sweep_host_to_host,
              r.best_host_workers,
              "maximum application-compute throughput with host quantization and argmax.",
              "disk/decode/resize end-to-end throughput.");

  os << "## 8. Long confirmation of selected maximum throughput\n\n"
     << "**How it works:** reruns the best worker counts from sections 6 and 7 for long sustained windows.\n\n"
     << "**Use to compare:** sustained platform capacity and thermal/power stability rather than short peaks.  \n"
     << "**Do not use for:** single-request latency.\n\n";
  if (!r.max_model_only.empty()) {
    os << "- Model-only, " << r.best_model_workers << " workers: **"
       << std::setprecision(2) << mean_field(r.max_model_only, &LatencyStats::fps) << " FPS**\n";
    write_power_md(os, aggregate_power_latency(r.max_model_only));
  } else os << "- Model-only confirmation skipped.\n\n";
  if (!r.max_host_to_host.empty()) {
    os << "- Host-to-host, " << r.best_host_workers << " workers: **"
       << std::setprecision(2) << mean_field(r.max_host_to_host, &LatencyStats::fps) << " FPS**\n";
    write_power_md(os, aggregate_power_latency(r.max_host_to_host));
  } else os << "- Host-to-host confirmation skipped.\n\n";

  os << "## 9. Fixed workload in memory - parallel\n\n"
     << "**How it works:** the same already-in-RAM real workload from section 3 is distributed across the best host-to-host worker count.\n\n"
     << "**Use to compare:** how quickly a finite in-memory workload finishes when DPU concurrency is exploited.  \n"
     << "**Do not use for:** disk/decode end-to-end behavior.\n\n"
     << "- Workers: **" << r.best_host_workers << "**\n"
     << "- Images/run: **" << workload_count << "**\n"
     << "- Mean completion time: **" << std::setprecision(6) << avg_workload_elapsed(r.fixed_memory_parallel) << " s**\n"
     << "- Mean throughput: **" << std::setprecision(2) << avg_workload_fps(r.fixed_memory_parallel) << " img/s**\n\n";
  write_power_md(os, aggregate_power_workloads(r.fixed_memory_parallel), workload_count);

  os << "## 10. Full dataset end-to-end worker sweep - PRIMARY REAL-WORKLOAD TEST\n\n"
     << "**How it works:** every labeled test image is processed from its file path to final class while the workload is distributed across 1, 2, 3 and 4 independent workers/runners. "
        "The wall timer starts immediately before the workers are released and stops only after the last image result is complete.\n\n"
     << "**Includes:** file read, decode, BGR->RGB, resize, quantization, parallel scheduling, VART/DPU inference, wait, argmax and accuracy bookkeeping.  \n"
     << "**Excludes:** initial XMODEL deserialize, runner creation and warmup.  \n"
     << "**Use to compare:** the realistic question 'I have N image files; how many seconds until all N results are ready using the platform as fast as possible?'. This is the preferred end-to-end throughput/capacity comparison across platforms.  \n"
     << "**Do not use for:** latency of one individual request.\n\n"
     << "Dataset completeness: **" << (dataset_count == 12630 ? "FULL OFFICIAL 12,630" : "INCOMPLETE - copy the official 12,630-image test set") << "**\n\n"
     << "| Workers | Mean total time (s) | Mean img/s | Accuracy | Projected 2000 images (s) | Mean W | Min W | Max W | 95% CI mean W | Mean energy/run (J) | J/image |\n"
     << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
  for (const auto& point : r.full_e2e_worker_sweep) {
    const double fps = point_mean_fps(point);
    const double elapsed = point_mean_elapsed(point);
    const double acc = point_mean_acc(point);
    const auto power = aggregate_power_workloads(point.runs);
    const uint64_t images = point.runs.empty() ? dataset_count : point.runs.front().images;
    os << "| " << point.workers << " | " << std::setprecision(6) << elapsed
       << " | " << std::setprecision(2) << fps
       << " | " << std::setprecision(2) << acc * 100.0 << "%"
       << " | " << std::setprecision(4) << (fps > 0.0 ? 2000.0 / fps : 0.0) << " | ";
    if (power.available) {
      os << std::setprecision(3) << power.mean_w << " | " << power.min_w << " | "
         << power.max_w << " | [" << power.ci95_low_w << ", " << power.ci95_high_w << "] | "
         << power.energy_j << " | "
         << (images > 0 ? power.energy_j / static_cast<double>(images) : 0.0) << " |\n";
    } else {
      os << "N/A | N/A | N/A | N/A | N/A | N/A |\n";
    }
  }
  os << "\nBest full end-to-end worker count by mean throughput: **" << r.best_full_e2e_workers << "**\n\n";

  os << "## 11. Cold start\n\n"
     << "**How it works:** fresh process -> XMODEL load -> runner creation -> image decode/preprocess -> first inference -> process exit.\n\n"
     << "**Use to compare:** application startup cost only.  \n"
     << "**Do not use for:** steady-state inference or throughput. Power is not reported here because the interval is too short and the current fork/exec path can be runtime-sensitive.\n\n";
  if (!r.cold_start_ms.empty()) {
    auto tmp = r.cold_start_ms;
    std::sort(tmp.begin(), tmp.end());
    const double mean = std::accumulate(tmp.begin(), tmp.end(), 0.0) / tmp.size();
    os << "- Mean: **" << std::setprecision(3) << mean << " ms**\n"
       << "- Median: **" << percentile_sorted(tmp, 50.0) << " ms**\n\n";
  } else os << "Skipped.\n\n";

  os << "## Recommended comparison mapping\n\n"
     << "- **Batch-1 latency:** section 1 (`single_stream_model_only`).\n"
     << "- **Batch-1 application compute:** section 2 (`single_stream_host_to_host`).\n"
     << "- **Finite in-memory workload:** sections 3 and 9.\n"
     << "- **Single-stream real file processing:** sections 4 and 5.\n"
     << "- **Maximum accelerator/application capacity without disk/decode:** sections 6-8.\n"
     << "- **Maximum real file-to-result capacity and actual time-to-finish N files:** **section 10**.\n"
     << "- **Startup cost:** section 11.\n";
}

int main(int argc, char** argv) {
  try {
    Options o = parse_args(argc, argv);
    if (o.cold_child) return cold_child_main(o);

    std::cout << "=============================================\n"
              << " TrafficSignNet ZCU104 C++ benchmark suite\n"
              << "=============================================\n";

    auto graph = xir::Graph::deserialize(o.model);
    auto* dpu = find_single_dpu_subgraph(graph.get());
    RunnerContext probe(dpu);
    QuantizerLUT quantizer(probe.input_fix());

    auto dataset = list_dataset(o.dataset);
    if (o.require_full_gtsrb_test) {
      if (dataset.size() != 12630) {
        std::ostringstream oss;
        oss << "Full GTSRB test required: expected exactly 12630 labeled images, found "
            << dataset.size() << " under " << o.dataset;
        throw std::runtime_error(oss.str());
      }
      std::array<bool, 43> class_seen{};
      for (const auto& ref : dataset) {
        if (ref.label < 0 || ref.label >= 43) {
          throw std::runtime_error("Full GTSRB test required: found label outside 0..42");
        }
        class_seen[static_cast<size_t>(ref.label)] = true;
      }
      for (size_t c = 0; c < class_seen.size(); ++c) {
        if (!class_seen[c]) {
          std::ostringstream oss;
          oss << "Full GTSRB test required: class " << c << " is missing";
          throw std::runtime_error(oss.str());
        }
      }
      std::cout << "Verified full official-style GTSRB test layout: 12630 images across labels 0..42\n";
    }
    auto workload_refs = select_workload(dataset, o.workload_size, o.seed);
    auto workload_mem = preload_images(workload_refs, probe.width(), probe.height());
    auto first_q = quantizer.quantize_copy(workload_mem.front().rgb);

    std::cout << "Model: " << o.model << "\n"
              << "Input: [1," << probe.height() << "," << probe.width() << ","
              << probe.channels() << "] fix=" << probe.input_fix() << "\n"
              << "Output elements: " << probe.output_elements() << " fix="
              << probe.output_fix() << "\n"
              << "Dataset images found: " << dataset.size() << "\n"
              << "Official GTSRB test complete: " << (dataset.size() == 12630 ? "YES" : "NO") << "\n"
              << "Fixed workload: " << workload_mem.size() << " images\n";
    {
      const auto p = detect_power_sensor_path();
      std::cout << "Power sensor: " << (p.empty() ? "not found" : p.string()) << "\n\n";
    }

    BenchmarkResults r;

    std::cout << "[1/11] Single-stream model-only\n";
    {
      RunnerContext ctx(dpu);
      r.single_model_only = benchmark_single_model_only(ctx, first_q, o.warmup,
                                                         o.runs, o.seconds,
                                                         o.power_sample_ms);
    }

    std::cout << "\n[2/11] Single-stream host-to-host\n";
    {
      RunnerContext ctx(dpu);
      r.single_host_to_host = benchmark_single_host(ctx, workload_mem.front().rgb,
                                                     quantizer, o.warmup,
                                                     o.runs, o.seconds,
                                                     o.power_sample_ms);
    }

    std::cout << "\n[3/11] Fixed workload in memory, single stream\n";
    {
      RunnerContext ctx(dpu);
      for (int i = 0; i < std::min(o.warmup, 200); ++i)
        (void)ctx.host_to_host(workload_mem.front().rgb, quantizer);
      for (int i = 0; i < o.workload_runs; ++i) {
        auto w = run_fixed_memory_single(ctx, workload_mem, quantizer, o.power_sample_ms);
        std::cout << "  run " << (i + 1) << ": " << std::fixed << std::setprecision(6)
                  << w.perf.elapsed_s << " s total | " << std::setprecision(2)
                  << w.perf.fps << " img/s | acc " << (w.accuracy * 100.0) << "%\n";
        r.fixed_memory_single.push_back(w);
      }
    }

    std::cout << "\n[4/11] Fixed workload end-to-end, single stream\n";
    {
      RunnerContext ctx(dpu);
      for (int i = 0; i < std::min(o.warmup, 100); ++i)
        (void)ctx.host_to_host(workload_mem.front().rgb, quantizer);
      for (int i = 0; i < o.workload_runs; ++i) {
        auto w = run_fixed_end_to_end_single(ctx, workload_refs, quantizer, o.power_sample_ms);
        std::cout << "  run " << (i + 1) << ": " << std::fixed << std::setprecision(6)
                  << w.perf.elapsed_s << " s total | " << std::setprecision(2)
                  << w.perf.fps << " img/s | acc " << (w.accuracy * 100.0) << "%\n";
        r.fixed_end_to_end_single.push_back(w);
      }
    }

    std::cout << "\n[5/11] All available dataset images end-to-end\n";
    if (!o.skip_full_test) {
      RunnerContext ctx(dpu);
      r.full_test_end_to_end = run_fixed_end_to_end_single(ctx, dataset, quantizer, o.power_sample_ms);
      r.full_test_ran = true;
      std::cout << "  " << r.full_test_end_to_end.images << " images | "
                << std::fixed << std::setprecision(2)
                << r.full_test_end_to_end.perf.fps << " img/s | acc "
                << r.full_test_end_to_end.accuracy * 100.0 << "%\n";
    } else {
      std::cout << "  skipped\n";
    }

    std::cout << "\n[6/11] Worker sweep model-only\n";
    r.sweep_model_only = run_worker_sweep(dpu, quantizer, workload_mem, first_q,
                                           o.max_workers, o.sweep_warmup,
                                           o.sweep_runs, o.sweep_seconds, true,
                                           o.power_sample_ms);
    r.best_model_workers = best_workers(r.sweep_model_only);
    std::cout << "  best model-only workers: " << r.best_model_workers << "\n";

    std::cout << "\n[7/11] Worker sweep host-to-host\n";
    r.sweep_host_to_host = run_worker_sweep(dpu, quantizer, workload_mem, first_q,
                                             o.max_workers, o.sweep_warmup,
                                             o.sweep_runs, o.sweep_seconds, false,
                                             o.power_sample_ms);
    r.best_host_workers = best_workers(r.sweep_host_to_host);
    std::cout << "  best host-to-host workers: " << r.best_host_workers << "\n";

    std::cout << "\n[8/11] Long confirmation of best worker counts\n";
    if (!o.skip_max_confirmation) {
      r.max_model_only = confirm_parallel(dpu, quantizer, workload_mem, first_q,
                                          r.best_model_workers, o.sweep_warmup,
                                          o.runs, o.seconds, true,
                                          o.power_sample_ms);
      r.max_host_to_host = confirm_parallel(dpu, quantizer, workload_mem, first_q,
                                            r.best_host_workers, o.sweep_warmup,
                                            o.runs, o.seconds, false,
                                            o.power_sample_ms);
    } else {
      std::cout << "  skipped\n";
    }

    std::cout << "\n[9/11] Fixed workload in memory, parallel\n";
    for (int i = 0; i < o.workload_runs; ++i) {
      auto w = run_fixed_memory_parallel(dpu, quantizer, workload_mem,
                                         r.best_host_workers, o.sweep_warmup,
                                         o.power_sample_ms);
      std::cout << "  run " << (i + 1) << " workers=" << r.best_host_workers
                << ": " << std::fixed << std::setprecision(6) << w.perf.elapsed_s
                << " s total | " << std::setprecision(2) << w.perf.fps << " img/s\n";
      r.fixed_memory_parallel.push_back(w);
    }

    std::cout << "\n[10/11] Full dataset end-to-end worker sweep\n";
    if (!o.skip_full_test) {
      r.full_e2e_worker_sweep = run_full_e2e_worker_sweep(
          dpu, quantizer, dataset, o.max_workers, o.sweep_warmup,
          o.full_e2e_runs, o.power_sample_ms);
      r.best_full_e2e_workers = best_workload_workers(r.full_e2e_worker_sweep);
      std::cout << "  best full end-to-end workers: " << r.best_full_e2e_workers << "\n";
    } else {
      std::cout << "  skipped because --skip-full-test was set\n";
    }

    std::cout << "\n[11/11] Cold start\n";
    if (!o.skip_cold) {
      const std::string exe = self_exe_path();
      for (int i = 0; i < o.cold_runs; ++i) {
        double ms = run_one_cold_start(exe, o, workload_refs.front().path);
        r.cold_start_ms.push_back(ms);
        std::cout << "  cold run " << (i + 1) << ": " << std::fixed
                  << std::setprecision(3) << ms << " ms\n";
      }
    } else {
      std::cout << "  skipped\n";
    }

    fs::create_directories(o.output_dir);
    const std::string stamp = now_timestamp();
    fs::path json_path = fs::path(o.output_dir) / ("zcu104_cpp_" + stamp + ".json");
    fs::path md_path = fs::path(o.output_dir) / ("zcu104_cpp_" + stamp + ".md");
    write_json(json_path, o, probe, dataset.size(), workload_mem.size(), r);
    write_markdown(md_path, o, dataset.size(), workload_mem.size(), r);

    std::cout << "\n=============================================\n"
              << "DONE\n"
              << "JSON: " << json_path << "\n"
              << "Markdown: " << md_path << "\n"
              << "=============================================\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
