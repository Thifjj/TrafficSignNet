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
  return w;
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
     << p << "  \"p999_ms\": " << s.p999_ms << "\n"
     << p << "}";
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
     << p << "  \"p999_ms_across_runs\": " << mean_field(runs, &LatencyStats::p999_ms) << "\n"
     << p << "}";
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

  os << "{\n"
     << "  \"schema_version\": 1,\n"
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
     << "    \"cold_runs\": " << o.cold_runs << "\n"
     << "  },\n"
     << "  \"benchmarks\": {\n";

  os << "    \"single_stream_model_only\": {\n";
  write_section_meta(os,
      "Synchronous batch-1 VART/DPU execution with an already quantized INT8 input resident in the runner input buffer.",
      "Primary latency benchmark and fair model-runtime reference for batch-1 comparisons.",
      {"VART execute_async", "blocking wait for completion"},
      {"disk I/O", "image decode", "resize", "RGB conversion", "normalization/quantization", "argmax", "model/runner creation"},
      {"One request is in flight at a time; this is latency-oriented, not maximum two-DPU-core throughput."}, 6);
  os << "      \"summary\": "; write_latency_summary(os, r.single_model_only, 6); os << ",\n"
     << "      \"runs\": "; write_runs(os, r.single_model_only, 6); os << "\n    },\n";

  os << "    \"single_stream_host_to_host\": {\n";
  write_section_meta(os,
      "Synchronous batch-1 application-compute path starting from a decoded/resized RGB uint8 image in host RAM.",
      "Measures host preprocessing overhead plus one DPU inference and class decision.",
      {"INT8 LUT quantization", "copy into runner input buffer", "VART execute_async", "blocking wait", "INT8 argmax"},
      {"disk I/O", "PNG/JPEG decode", "resize", "model/runner creation"},
      {"Uses an optimized 256-entry LUT that is mathematically equivalent to round((pixel/255)*2^input_fix_point)."}, 6);
  os << "      \"summary\": "; write_latency_summary(os, r.single_host_to_host, 6); os << ",\n"
     << "      \"runs\": "; write_runs(os, r.single_host_to_host, 6); os << "\n    },\n";

  os << "    \"fixed_workload_memory_single_stream\": {\n";
  write_section_meta(os,
      "Closed workload of different real dataset images already decoded/resized and resident in RAM, processed serially at batch 1.",
      "Answers: if N real images are already in memory, how long until the Nth result is complete on a single synchronous stream?",
      {"per-image INT8 LUT quantization", "DPU inference", "wait", "argmax", "accuracy bookkeeping"},
      {"disk I/O", "decode", "resize", "model/runner creation"},
      {"Total elapsed time is measured directly from the first image start to completion of the last image; FPS is images/that wall time."}, 6);
  os << "      \"runs\": "; write_workload_runs(os, r.fixed_memory_single, 6); os << "\n    },\n";

  os << "    \"fixed_workload_end_to_end_single_stream\": {\n";
  write_section_meta(os,
      "Closed workload of different real image files processed serially from filesystem path to predicted class.",
      "Closest local-file measurement to 'I gave the application N image files; when was the last result ready?'.",
      {"file read", "image decode", "BGR->RGB", "resize", "INT8 LUT quantization", "DPU inference", "wait", "argmax"},
      {"model/runner creation"},
      {"OS page cache is not flushed between repetitions, so later runs can benefit from cached file data."}, 6);
  os << "      \"runs\": "; write_workload_runs(os, r.fixed_end_to_end_single, 6); os << "\n    },\n";

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

