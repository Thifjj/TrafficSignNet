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

static void write_markdown(const fs::path& path, const Options& o,
                           size_t dataset_count, size_t workload_count,
                           const BenchmarkResults& r) {
  std::ofstream os(path);
  if (!os) throw std::runtime_error("Cannot create Markdown: " + path.string());
  os << std::fixed;
  os << "# TrafficSignNet ZCU104 C++ benchmark report\n\n"
     << "Model: `" << o.model << "`  \n"
     << "Dataset: `" << o.dataset << "` (" << dataset_count << " labeled images found)  \n"
     << "Fixed workload: " << workload_count << " different images  \n"
     << "Batch size: **1**\n\n";

  os << "## 1. Single-stream model-only\n\n"
     << "Timed: already-quantized INT8 input -> VART `execute_async` -> `wait`. "
     << "No preprocessing or argmax. This is the primary batch-1 latency result.\n\n"
     << "- Mean FPS across runs: **" << std::setprecision(2)
     << mean_field(r.single_model_only, &LatencyStats::fps) << "**\n"
     << "- Mean latency: **" << std::setprecision(4)
     << mean_field(r.single_model_only, &LatencyStats::mean_ms) << " ms**\n"
     << "- Mean P99: **" << mean_field(r.single_model_only, &LatencyStats::p99_ms) << " ms**\n\n";

  os << "## 2. Single-stream host-to-host\n\n"
     << "Starts from decoded/resized RGB uint8 in RAM. Timed: LUT INT8 quantization + DPU + wait + argmax.\n\n"
     << "- Mean FPS: **" << std::setprecision(2)
     << mean_field(r.single_host_to_host, &LatencyStats::fps) << "**\n"
     << "- Mean latency: **" << std::setprecision(4)
     << mean_field(r.single_host_to_host, &LatencyStats::mean_ms) << " ms**\n\n";

  os << "## 3. Fixed workload in memory, single stream\n\n"
     << "Different real images are preloaded to RAM before timing. The important number is total wall time until the last result.\n\n"
     << "- Images/workload: **" << workload_count << "**\n"
     << "- Mean total completion time: **" << std::setprecision(6)
     << avg_workload_elapsed(r.fixed_memory_single) << " s**\n"
     << "- Mean throughput: **" << std::setprecision(2)
     << avg_workload_fps(r.fixed_memory_single) << " img/s**\n"
     << "- Mean accuracy: **" << std::setprecision(2)
     << avg_workload_acc(r.fixed_memory_single) * 100.0 << "%**\n\n";

  os << "## 4. Fixed workload file-to-result, single stream\n\n"
     << "Includes file read, decode, RGB conversion, resize, quantization, DPU and argmax. OS page cache is not flushed.\n\n"
     << "- Mean total completion time: **" << std::setprecision(6)
     << avg_workload_elapsed(r.fixed_end_to_end_single) << " s**\n"
     << "- Mean throughput: **" << std::setprecision(2)
     << avg_workload_fps(r.fixed_end_to_end_single) << " img/s**\n"
     << "- Mean accuracy: **" << std::setprecision(2)
     << avg_workload_acc(r.fixed_end_to_end_single) * 100.0 << "%**\n\n";

  os << "## 5. All available dataset images end-to-end\n\n";
  if (r.full_test_ran) {
    os << "- Images: **" << r.full_test_end_to_end.images << "**\n"
       << "- Accuracy: **" << std::setprecision(2)
       << r.full_test_end_to_end.accuracy * 100.0 << "%**\n"
       << "- Total time: **" << std::setprecision(6)
       << r.full_test_end_to_end.perf.elapsed_s << " s**\n"
       << "- Throughput: **" << std::setprecision(2)
       << r.full_test_end_to_end.perf.fps << " img/s**\n\n";
  } else {
    os << "Skipped.\n\n";
  }
  os << "> Note: this processes every labeled image actually copied to the board. The repository deployment subset is not the full official 12,630-image GTSRB test split.\n\n";

  auto sweep_table = [&](const char* title, const std::vector<SweepPoint>& sweep, int best) {
    os << "## " << title << "\n\n"
       << "Independent VART runners are used concurrently. Runner creation and warmup are outside timing.\n\n"
       << "| Workers | Mean FPS | Mean latency/request (ms) |\n"
       << "|---:|---:|---:|\n";
    for (const auto& p : sweep) {
      os << "| " << p.workers << " | " << std::setprecision(2)
         << mean_field(p.runs, &LatencyStats::fps) << " | " << std::setprecision(4)
         << mean_field(p.runs, &LatencyStats::mean_ms) << " |\n";
    }
    os << "\nSelected best worker count: **" << best << "**\n\n";
  };
  sweep_table("6. Worker sweep - model-only maximum throughput", r.sweep_model_only,
              r.best_model_workers);
  sweep_table("7. Worker sweep - host-to-host maximum throughput", r.sweep_host_to_host,
              r.best_host_workers);

  os << "## 8. Confirmed maximum sustained throughput\n\n";
  if (!r.max_model_only.empty()) {
    os << "- Model-only, " << r.best_model_workers << " workers: **"
       << std::setprecision(2) << mean_field(r.max_model_only, &LatencyStats::fps)
       << " FPS**\n";
  } else os << "- Model-only confirmation skipped.\n";
  if (!r.max_host_to_host.empty()) {
    os << "- Host-to-host, " << r.best_host_workers << " workers: **"
       << std::setprecision(2) << mean_field(r.max_host_to_host, &LatencyStats::fps)
       << " FPS**\n\n";
  } else os << "- Host-to-host confirmation skipped.\n\n";

  os << "## 9. Fixed workload in memory, parallel\n\n"
     << "Uses the best host-to-host worker count from the sweep. This is the direct 'how long until all "
     << workload_count << " images are done when concurrency is enabled?' measurement.\n\n"
     << "- Workers: **" << r.best_host_workers << "**\n"
     << "- Mean completion time: **" << std::setprecision(6)
     << avg_workload_elapsed(r.fixed_memory_parallel) << " s**\n"
     << "- Mean throughput: **" << std::setprecision(2)
     << avg_workload_fps(r.fixed_memory_parallel) << " img/s**\n\n";

  os << "## 10. Cold start\n\n"
     << "Fresh process -> XMODEL load -> runner creation -> image decode/preprocess -> first inference -> exit.\n\n";
  if (!r.cold_start_ms.empty()) {
    auto tmp = r.cold_start_ms;
    std::sort(tmp.begin(), tmp.end());
    double mean = std::accumulate(tmp.begin(), tmp.end(), 0.0) / tmp.size();
    os << "- Mean: **" << std::setprecision(3) << mean << " ms**\n"
       << "- Median: **" << percentile_sorted(tmp, 50.0) << " ms**\n\n";
  } else os << "Skipped.\n\n";

  os << "## How to use these numbers\n\n"
     << "- Compare **single-stream model-only batch 1** across CPU/GPU/DPU/HLS4ML when discussing latency.\n"
     << "- Compare **single-stream host-to-host** when discussing application compute overhead.\n"
     << "- Use **fixed workload** when saying exactly how long 100 real images took to finish.\n"
     << "- Use **worker sweep / confirmed max throughput** only when discussing platform capacity; do not substitute it for single-request latency.\n"
     << "- Use **end-to-end** when file ingestion/decoding is part of the application scenario.\n";
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
    auto workload_refs = select_workload(dataset, o.workload_size, o.seed);
    auto workload_mem = preload_images(workload_refs, probe.width(), probe.height());
    auto first_q = quantizer.quantize_copy(workload_mem.front().rgb);

    std::cout << "Model: " << o.model << "\n"
              << "Input: [1," << probe.height() << "," << probe.width() << ","
              << probe.channels() << "] fix=" << probe.input_fix() << "\n"
              << "Output elements: " << probe.output_elements() << " fix="
              << probe.output_fix() << "\n"
              << "Dataset images found: " << dataset.size() << "\n"
              << "Fixed workload: " << workload_mem.size() << " images\n\n";

    BenchmarkResults r;

    std::cout << "[1/10] Single-stream model-only\n";
    {
      RunnerContext ctx(dpu);
      r.single_model_only = benchmark_single_model_only(ctx, first_q, o.warmup,
                                                         o.runs, o.seconds);
    }

    std::cout << "\n[2/10] Single-stream host-to-host\n";
    {
      RunnerContext ctx(dpu);
      r.single_host_to_host = benchmark_single_host(ctx, workload_mem.front().rgb,
                                                     quantizer, o.warmup,
                                                     o.runs, o.seconds);
    }

    std::cout << "\n[3/10] Fixed workload in memory, single stream\n";
    {
      RunnerContext ctx(dpu);
      for (int i = 0; i < std::min(o.warmup, 200); ++i)
        (void)ctx.host_to_host(workload_mem.front().rgb, quantizer);
      for (int i = 0; i < o.workload_runs; ++i) {
        auto w = run_fixed_memory_single(ctx, workload_mem, quantizer);
        std::cout << "  run " << (i + 1) << ": " << std::fixed << std::setprecision(6)
                  << w.perf.elapsed_s << " s total | " << std::setprecision(2)
                  << w.perf.fps << " img/s | acc " << (w.accuracy * 100.0) << "%\n";
        r.fixed_memory_single.push_back(w);
      }
    }

    std::cout << "\n[4/10] Fixed workload end-to-end, single stream\n";
    {
      RunnerContext ctx(dpu);
      for (int i = 0; i < std::min(o.warmup, 100); ++i)
        (void)ctx.host_to_host(workload_mem.front().rgb, quantizer);
      for (int i = 0; i < o.workload_runs; ++i) {
        auto w = run_fixed_end_to_end_single(ctx, workload_refs, quantizer);
        std::cout << "  run " << (i + 1) << ": " << std::fixed << std::setprecision(6)
                  << w.perf.elapsed_s << " s total | " << std::setprecision(2)
                  << w.perf.fps << " img/s | acc " << (w.accuracy * 100.0) << "%\n";
        r.fixed_end_to_end_single.push_back(w);
      }
    }

    std::cout << "\n[5/10] All available dataset images end-to-end\n";
    if (!o.skip_full_test) {
      RunnerContext ctx(dpu);
      r.full_test_end_to_end = run_fixed_end_to_end_single(ctx, dataset, quantizer);
      r.full_test_ran = true;
      std::cout << "  " << r.full_test_end_to_end.images << " images | "
                << std::fixed << std::setprecision(2)
                << r.full_test_end_to_end.perf.fps << " img/s | acc "
                << r.full_test_end_to_end.accuracy * 100.0 << "%\n";
    } else {
      std::cout << "  skipped\n";
    }

    std::cout << "\n[6/10] Worker sweep model-only\n";
    r.sweep_model_only = run_worker_sweep(dpu, quantizer, workload_mem, first_q,
                                           o.max_workers, o.sweep_warmup,
                                           o.sweep_runs, o.sweep_seconds, true);
    r.best_model_workers = best_workers(r.sweep_model_only);
    std::cout << "  best model-only workers: " << r.best_model_workers << "\n";

    std::cout << "\n[7/10] Worker sweep host-to-host\n";
    r.sweep_host_to_host = run_worker_sweep(dpu, quantizer, workload_mem, first_q,
                                             o.max_workers, o.sweep_warmup,
                                             o.sweep_runs, o.sweep_seconds, false);
    r.best_host_workers = best_workers(r.sweep_host_to_host);
    std::cout << "  best host-to-host workers: " << r.best_host_workers << "\n";

    std::cout << "\n[8/10] Long confirmation of best worker counts\n";
    if (!o.skip_max_confirmation) {
      r.max_model_only = confirm_parallel(dpu, quantizer, workload_mem, first_q,
                                          r.best_model_workers, o.sweep_warmup,
                                          o.runs, o.seconds, true);
      r.max_host_to_host = confirm_parallel(dpu, quantizer, workload_mem, first_q,
                                            r.best_host_workers, o.sweep_warmup,
                                            o.runs, o.seconds, false);
    } else {
      std::cout << "  skipped\n";
    }

    std::cout << "\n[9/10] Fixed workload in memory, parallel\n";
    for (int i = 0; i < o.workload_runs; ++i) {
      auto w = run_fixed_memory_parallel(dpu, quantizer, workload_mem,
                                         r.best_host_workers, o.sweep_warmup);
      std::cout << "  run " << (i + 1) << " workers=" << r.best_host_workers
                << ": " << std::fixed << std::setprecision(6) << w.perf.elapsed_s
                << " s total | " << std::setprecision(2) << w.perf.fps << " img/s\n";
      r.fixed_memory_parallel.push_back(w);
    }

    std::cout << "\n[10/10] Cold start\n";
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
