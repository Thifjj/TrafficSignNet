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
    int runs, double seconds) {
  ctx.load_prequantized(input);
  for (int i = 0; i < warmup; ++i) ctx.execute_only();

  std::vector<LatencyStats> out;
  for (int r = 0; r < runs; ++r) {
    std::vector<double> lat;
    lat.reserve(1000000);
    uint64_t count = 0;
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
    out.push_back(make_stats(std::move(lat), count, elapsed));
    std::cout << "  model-only run " << (r + 1) << ": " << std::fixed
              << std::setprecision(2) << out.back().fps << " FPS | "
              << std::setprecision(4) << out.back().mean_ms << " ms\n";
  }
  return out;
}

static std::vector<LatencyStats> benchmark_single_host(
    RunnerContext& ctx, const std::vector<uint8_t>& rgb,
    const QuantizerLUT& quantizer, int warmup, int runs, double seconds) {
  for (int i = 0; i < warmup; ++i) (void)ctx.host_to_host(rgb, quantizer);

  std::vector<LatencyStats> out;
  for (int r = 0; r < runs; ++r) {
    std::vector<double> lat;
    lat.reserve(1000000);
    uint64_t count = 0;
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
    out.push_back(make_stats(std::move(lat), count, elapsed));
    std::cout << "  host-to-host run " << (r + 1) << ": " << std::fixed
              << std::setprecision(2) << out.back().fps << " FPS | "
              << std::setprecision(4) << out.back().mean_ms << " ms\n";
  }
  return out;
}

static WorkloadRun run_fixed_memory_single(RunnerContext& ctx,
                                           const std::vector<MemoryImage>& images,
                                           const QuantizerLUT& q) {
  std::vector<double> lat;
  lat.reserve(images.size());
  uint64_t correct = 0;
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
  return w;
}

static WorkloadRun run_fixed_end_to_end_single(RunnerContext& ctx,
                                                const std::vector<ImageRef>& refs,
                                                const QuantizerLUT& q) {
  std::vector<double> lat;
  lat.reserve(refs.size());
  uint64_t correct = 0;
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
  return w;
}

static LatencyStats run_parallel_sustained(const xir::Subgraph* dpu,
                                            const QuantizerLUT& q,
                                            const std::vector<MemoryImage>& images,
                                            const std::vector<int8_t>& prequantized,
                                            int workers, int warmup_per_worker,
                                            double seconds, bool model_only) {
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
  return make_stats(std::move(merged), total_count, elapsed);
}

static std::vector<SweepPoint> run_worker_sweep(const xir::Subgraph* dpu,
                                                 const QuantizerLUT& q,
                                                 const std::vector<MemoryImage>& images,
                                                 const std::vector<int8_t>& prequantized,
                                                 int max_workers,
                                                 int warmup_per_worker,
                                                 int runs, double seconds,
                                                 bool model_only) {
  std::vector<SweepPoint> sweep;
  for (int workers = 1; workers <= max_workers; ++workers) {
    SweepPoint p;
    p.workers = workers;
    std::cout << "  workers=" << workers << (model_only ? " model-only" : " host-to-host") << "\n";
    for (int r = 0; r < runs; ++r) {
      auto result = run_parallel_sustained(dpu, q, images, prequantized,
                                           workers, warmup_per_worker,
                                           seconds, model_only);
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
                                                   bool model_only) {
  std::vector<LatencyStats> out;
  for (int r = 0; r < runs; ++r) {
    auto x = run_parallel_sustained(dpu, q, images, prequantized, workers,
                                    warmup, seconds, model_only);
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
                                              int workers, int warmup_per_worker) {
  std::vector<std::unique_ptr<RunnerContext>> contexts;
  contexts.reserve(workers);
  for (int i = 0; i < workers; ++i) contexts.push_back(std::make_unique<RunnerContext>(dpu));
