#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
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
  bool skip_full_test = false;
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
      << "  --seed N                     Deterministic workload seed (default 42)\n"
      << "  --skip-full-test             Skip all-images end-to-end test\n"
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
    } else if (a == "--skip-full-test") {
      o.skip_full_test = true;
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
      o.parallel_workers <= 0 || o.cold_runs <= 0) {
    throw std::runtime_error("Benchmark numeric arguments must be positive (warmups may be zero)");
  }
  return o;
}

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
