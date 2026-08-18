# TrafficSignNet ZCU104 C++ Benchmark

Suite C++ para medir latência e throughput do `trafficsignnet_int8.xmodel` diretamente via XIR/VART na ZCU104.

## Onde fica

```text
zcu104_inference/
├── model/trafficsignnet_int8.xmodel
├── dataset/
├── cpp/
│   ├── benchmark_zcu104.cpp
│   ├── benchmark_parts/
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── run_quick.sh
│   └── run_full.sh
└── results_cpp/
```

`benchmark_zcu104.cpp` apenas inclui os quatro blocos em `benchmark_parts/`. Eles são partes contíguas do mesmo fonte C++ e são recompostos pelo pré-processador durante a compilação.

## Antes de compilar

Na ZCU104:

```bash
xdputil query
xdputil xmodel ../model/trafficsignnet_int8.xmodel -l
g++ --version
cmake --version
```

## Compilar

```bash
cd /home/root/zcu104_inference/cpp
chmod +x build.sh run_quick.sh run_full.sh
./build.sh
```

O executável será:

```text
build/benchmark_zcu104
```

## Smoke test

```bash
./run_quick.sh
```

Use apenas para verificar XMODEL, VART, OpenCV, criação de múltiplos runners e geração de saída. Não use esses números no relatório final.

## Benchmark completo

```bash
./run_full.sh
```

Configuração padrão séria:

- batch 1;
- warmup de 1000 inferências;
- 5 runs de 60 s para steady state;
- workload fechado de 100 imagens, repetido 10 vezes;
- sweep de 1, 2, 3 e 4 workers/runners;
- 3 runs de 10 s por ponto do sweep;
- confirmação longa da melhor configuração;
- 3 cold starts.

## Benchmarks medidos

### 1. `single_stream_model_only`

Entrada INT8 já quantizada e residente no buffer. Cronometra `execute_async()` + `wait()`. Exclui decode, resize, quantização e argmax. É o principal teste de latência batch-1 do DPU.

### 2. `single_stream_host_to_host`

Começa de RGB uint8 já em RAM. Inclui quantização INT8 por LUT de 256 entradas, inferência, espera e argmax. Exclui leitura do arquivo e resize.

### 3. `fixed_workload_memory_single_stream`

Processa 100 imagens reais diferentes já carregadas na RAM. Mede diretamente o tempo entre o início da primeira e o término da centésima inferência.

### 4. `fixed_workload_end_to_end_single_stream`

Para cada imagem inclui leitura do arquivo, decode, BGR->RGB, resize, quantização, DPU e argmax. O cache de páginas do Linux não é limpo entre repetições.

### 5. `full_dataset_end_to_end`

Processa todas as imagens rotuladas presentes em `--dataset`, medindo accuracy, tempo total, throughput e latências. O deployment atual do repositório contém apenas um subconjunto; para o GTSRB test oficial completo copie as 12.630 imagens para a placa e aponte `--dataset` para ele.

### 6. `worker_sweep_model_only`

Cria runners independentes e mede 1..4 workers concorrentes. Serve para descobrir a configuração que melhor mantém os cores DPU ocupados.

### 7. `worker_sweep_host_to_host`

Mesmo sweep, mas inclui a quantização LUT e argmax em cada inferência. Mostra quando o host ARM passa a limitar o throughput.

### 8. `max_throughput_model_only` e `max_throughput_host_to_host`

Após o sweep, a suíte seleciona automaticamente o número de workers com maior FPS médio e repete uma medição longa nessa configuração.

### 9. `fixed_workload_memory_parallel`

Usa o melhor número de workers do host-to-host e processa o workload fechado de 100 imagens concorrentemente. Este é o teste principal para responder: "quanto tempo levou para terminar as 100 imagens usando a capacidade paralela disponível?"

### 10. `cold_start`

Mede processo novo -> carregamento do XMODEL -> criação do runner -> decode/preprocess da primeira imagem -> primeira inferência -> saída do processo. Fica separado do steady state.

## Saídas

Cada execução gera em `../results_cpp/`:

```text
zcu104_cpp_YYYYMMDD_HHMMSS.json
zcu104_cpp_YYYYMMDD_HHMMSS.md
```

O JSON é a fonte canônica. Cada seção explica `description`, `purpose`, o que entra e o que fica fora da região cronometrada, além das ressalvas metodológicas.

## Comparação correta

Para CPU/GPU/DPU/HLS4ML, use `single_stream_model_only`, batch 1, como comparação principal de latência. Use `single_stream_host_to_host` para custo computacional da aplicação. Use o workload fechado para dizer quanto 100 imagens realmente demoraram. Use o sweep/máximo throughput apenas como capacidade máxima da plataforma, sem substituir a latência de uma requisição.
