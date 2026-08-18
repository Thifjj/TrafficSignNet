# TrafficSignNet — documentação completa do projeto

## 1. Objetivo

Este projeto prepara e avalia uma rede neural de classificação de placas de
trânsito GTSRB em três formas:

1. modelo TensorFlow/Keras FP32;
2. modelo TensorFlow/Keras quantizado INT8 pelo Vitis AI;
3. modelo XMODEL compilado para o DPU `DPUCZDX8G` da ZCU104.

O fluxo completo é:

```text
TrafficSignNet FP32
        |
        | reconstrução compatível com TensorFlow 2.12
        v
TrafficSignNet_FP32_TF212.h5
        |
        | calibração/quantização com Vitis AI
        v
TrafficSignNet_INT8.h5
        |
        | vai_c_tensorflow2 + arch.json da ZCU104
        v
trafficsignnet_int8.xmodel
        |
        | VART/XIR no Linux da placa
        v
Inferência no DPU da ZCU104
```

## 2. Modelo

A entrada é uma imagem RGB normalizada para `[0,1]`, com formato NHWC:

```text
batch x altura x largura x canais = 1 x 32 x 32 x 3
```

A saída contém 43 logits, um para cada classe do GTSRB:

```text
1 x 43
```

A classe prevista é o índice do maior logit (`argmax`). Os diretórios do
dataset usam os IDs `00000` até `00042`.

A arquitetura definida em `scripts/rebuild_model_tf212.py` é:

```text
Entrada 32x32x3
Conv2D(16) + BatchNorm + ReLU + MaxPool
Conv2D(32) + BatchNorm + ReLU + MaxPool
Conv2D(64) + BatchNorm + ReLU + MaxPool
Conv2D(64) + BatchNorm + ReLU
Flatten
Dense(64, ReLU)
Dense(43 logits)
```

Não se aplica `softmax` para escolher a classe: `argmax(logits)` produz a mesma
classe e evita trabalho desnecessário.

## 3. Estrutura do projeto

```text
TrafficSignNet/
├── models/
│   ├── keras/
│   └── compiled/trafficsignnet_zcu104/
├── scripts/
│   ├── rebuild_model_tf212.py
│   ├── quantize_vai.py
│   └── package_zcu104.sh
├── benchmarks/
│   ├── tensorflow/
│   └── vitisai/
├── deploy/zcu104/
│   ├── cpp/
│   ├── model/
│   └── dataset/                 # local, ignorado pelo Git
├── data/gtsrb/                  # local, ignorado pelo Git
├── results/
├── README.md
└── DOCUMENTACAO.md
```

### Arquivos de modelo

| Arquivo | Finalidade |
|---|---|
| `models/keras/TrafficSignNet_FP32.h5` | Modelo Keras original em ponto flutuante. |
| `models/keras/TrafficSignNet_FP32_weights.npz` | Pesos portáveis, com metadados e checksums, usados para reconstruir o modelo em outra versão do TensorFlow. |
| `models/keras/TrafficSignNet_FP32_TF212.h5` | Modelo FP32 reconstruído no TensorFlow 2.12. É a entrada de `scripts/quantize_vai.py`. |
| `models/keras/TrafficSignNet_INT8.h5` | Modelo Keras quantizado pelo Vitis AI. Ainda é usado no host TensorFlow; não é o arquivo executado diretamente pelo DPU. |
| `models/compiled/trafficsignnet_zcu104/trafficsignnet_int8.xmodel` | Modelo final compilado para o DPU padrão da ZCU104. |

### Scripts do host

| Arquivo | Finalidade |
|---|---|
| `scripts/rebuild_model_tf212.py` | Recria a arquitetura, valida cada tensor do NPZ e salva um H5 compatível com TensorFlow 2.12. |
| `scripts/quantize_vai.py` | Usa imagens de validação para calibrar e gerar o modelo INT8. Usa 32 lotes de 32 imagens, aproximadamente 1.024 amostras. |
| `benchmarks/tensorflow/benchmark_cpu_model_and_host.py` | Referência de benchmark do modelo FP32 na CPU. |
| `benchmarks/vitisai/benchmark_cpu_model_and_host.py` | Benchmark do H5 quantizado na CPU usando as camadas customizadas do Vitis AI. |
| `benchmarks/vitisai/benchmark_nvidia_gpu.py` | Benchmark do H5 quantizado em uma GPU NVIDIA visível pelo TensorFlow. Não mede o DPU. |

### Arquivos compilados

`models/compiled/trafficsignnet_zcu104/meta.json` informa que o XMODEL usa:

```text
runner:  libvart-dpu-runner.so
kernel:  subgraph_quant_conv2d_1
target:  DPUCZDX8G_ISA1_B4096
```

`md5sum.txt` contém o MD5 gerado pelo compilador. Ele pode ser usado para
verificar se o arquivo não foi corrompido durante a cópia.

### Dataset

| Diretório | Uso | Tamanho aproximado |
|---|---|---:|
| `data/gtsrb/split/train` | Treinamento original | 226 MB |
| `data/gtsrb/split/val` | Validação e calibração INT8 | 52 MB |
| `data/gtsrb/split/test` | Avaliação completa | 90 MB |
| `deploy/zcu104/dataset` | Subconjunto leve para a placa: 10 imagens de cada uma das 43 classes | 3,4 MB aproximadamente |

O rótulo é obtido do nome da pasta. Por exemplo, uma imagem em `00014/` tem
classe esperada 14. O pacote da placa tem 430 imagens no total.

## 4. Reconstrução e quantização

Estes passos são necessários apenas ao recriar o modelo. Devem ser executados
no ambiente TensorFlow 2 do Vitis AI.

### Reconstruir o FP32 no TensorFlow 2.12

```bash
python3 scripts/rebuild_model_tf212.py
```

Entrada:

```text
models/keras/TrafficSignNet_FP32_weights.npz
```

Saída:

```text
models/keras/TrafficSignNet_FP32_TF212.h5
```

### Quantizar

```bash
python3 scripts/quantize_vai.py
```

O script normaliza as imagens com `/255.0`. Essa normalização deve continuar
igual durante calibração e inferência.

Saída:

```text
models/keras/TrafficSignNet_INT8.h5
```

## 5. Compilação para a ZCU104

Dentro do container TensorFlow 2 do Vitis AI 3.5:

```bash
mkdir -p models/compiled/trafficsignnet_zcu104

vai_c_tensorflow2 \
  --model models/keras/TrafficSignNet_INT8.h5 \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  --output_dir models/compiled/trafficsignnet_zcu104 \
  --net_name trafficsignnet_int8
```

O `arch.json` deve corresponder ao bitstream/DPU instalado na placa. Este
projeto usa a arquitetura padrão ZCU104 `DPUCZDX8G_ISA1_B4096`. Um bitstream
customizado pode exigir outro `arch.json` e uma nova compilação.

## 6. Pacote da ZCU104

`trafficsignnet_zcu104.tar.gz` é o arquivo único para transferência. Ele contém:

- XMODEL;
- programa de inferência VART;
- verificação do DPU;
- dataset reduzido e rotulado;
- benchmark e avaliação de acurácia;
- instruções locais.

### Transferir para a placa

No computador host:

```bash
scp trafficsignnet_zcu104.tar.gz root@IP_DA_ZCU104:/home/root/
```

Na placa:

```bash
cd /home/root
tar -xzf trafficsignnet_zcu104.tar.gz
cd zcu104
./check_board.sh
```

`check_board.sh` verifica `xdputil`, OpenCV, NumPy, VART e XIR; depois mostra o
DPU presente na placa e os subgrafos do XMODEL.

### Comandos de execução

Executar acurácia e os dois benchmarks com os padrões:

```bash
./run.sh
```

Teste rápido:

```bash
./run.sh --accuracy --benchmark --warmup 5 --runs 1 --seconds 2
```

Acurácia apenas:

```bash
./run.sh --accuracy
```

Benchmark completo apenas:

```bash
./run.sh --benchmark --warmup 1000 --runs 5 --seconds 60
```

Uma imagem:

```bash
./run.sh --image "$(find dataset/00014 -type f | sort | head -n 1)"
```

Por padrão, a saída é salva em `results.json`. Outro caminho pode ser usado:

```bash
./run.sh --benchmark --results results/minha_medicao.json
```

## 7. O que `inference.py` faz

1. Desserializa o XMODEL com `xir.Graph.deserialize`.
2. Procura o subgrafo marcado como `DPU`.
3. Cria um `vart.Runner` em modo `run`.
4. Descobre automaticamente formas e `fix_point` dos tensores.
5. Lê a imagem com OpenCV, converte BGR para RGB e redimensiona para 32x32.
6. Normaliza `uint8` para `float32` em `[0,1]`.
7. Quantiza para INT8 usando o `fix_point` do tensor de entrada.
8. Executa o DPU com `execute_async` e aguarda com `wait`.
9. Obtém a classe com `argmax`.

No XMODEL atual, os tensores detectados na placa foram:

```text
entrada: quant_image       [1, 32, 32, 3], fix_point = 6
saída:   quant_logits_fix  [1, 43],         fix_point = 1
```

Para a entrada, `fix_point=6` significa aproximadamente:

```text
int8 = round((uint8 / 255.0) * 2^6)
```

O código lê esse valor do modelo em vez de fixá-lo manualmente.

## 8. Definição dos dois benchmarks

### Model-only

Escopo:

```text
entrada INT8 já preparada
→ submissão ao VART
→ execução DPU
→ sincronização/saída
```

Não inclui leitura de imagem, resize, normalização, quantização nem `argmax`.
Serve para medir o custo do modelo/runner no acelerador.

### Host-to-host

Escopo:

```text
imagem RGB uint8 32x32 já em memória
→ conversão float32
→ normalização /255
→ quantização INT8
→ execução e sincronização DPU
→ argmax
```

Não inclui acesso ao disco nem resize. Isso mantém o significado equivalente ao
benchmark TensorFlow original, que também começa com um tensor `uint8` já em
memória.

Os dois modos são síncronos, batch 1 e single-thread no lado do chamador. Eles
medem latência, não throughput máximo com vários runners/requisições paralelas.

## 9. Significado das métricas

| Campo | Significado |
|---|---|
| `warmup` | Inferências descartadas antes da medição para estabilizar caches e runtime. |
| `runs` | Quantidade de janelas independentes de medição. |
| `seconds_per_run` | Duração alvo de cada janela. |
| `inferences` | Inferências concluídas naquela janela. |
| `elapsed_seconds` | Tempo real da janela. |
| `fps` | Inferências por segundo: `inferences / elapsed_seconds`. Com batch 1, FPS também é imagens/s. |
| `mean` | Latência média. É sensível a pausas e outliers. |
| `median` | Percentil 50; metade das inferências foi mais rápida. |
| `min` / `max` | Menor e maior latência observadas. |
| `p90` | 90% das inferências terminaram até esse tempo. |
| `p95` | 95% terminaram até esse tempo. |
| `p99` | 99% terminaram até esse tempo. Útil para observar cauda de latência. |
| `p99_9` | 99,9% terminaram até esse tempo; evidencia eventos raros. |
| `fps_mean` | Média do FPS das execuções. |
| `latency_mean_ms` | Média das latências médias de cada execução. |

Não se deve calcular FPS simplesmente como `1000 / latency_mean_ms` quando há
overhead de laço e janela de tempo; o script calcula FPS usando contagem e tempo
real.

## 10. Resultados atuais

### ZCU104 — DPU

Arquivo: `results/zcu104/results.json`.

Configuração registrada:

```text
Python:       3.9.9
batch:        1
warmup:       1.000 inferências por modo
runs:         5 por modo
tempo:        60 segundos por run
tempo total:  aproximadamente 10 minutos, sem contar warmup
```

| Modo | FPS médio | Latência média |
|---|---:|---:|
| Model-only | 4.498,84 FPS | 0,2158 ms |
| Host-to-host | 1.539,41 FPS | 0,6431 ms |

O model-only foi muito estável: as cinco execuções ficaram aproximadamente
entre 4.491 e 4.509 FPS, com P99 ao redor de 0,228 ms.

No host-to-host, as cinco execuções ficaram aproximadamente entre 1.538 e 1.541
FPS, com P99 entre 0,664 e 0,670 ms.

A diferença de aproximadamente 0,427 ms mostra que, para uma rede pequena, a
normalização e a quantização feitas em NumPy/Python dominam boa parte do tempo
host-to-host. Isso não significa que o DPU ficou mais lento; significa que o
modelo é rápido o bastante para o processamento do host se tornar relevante.

Esse arquivo da ZCU104 **não possui o campo `accuracy`**, pois a execução que o
gerou usou somente `--benchmark`. Para registrar acurácia junto aos benchmarks:

```bash
./run.sh --accuracy --benchmark --warmup 1000 --runs 5 --seconds 60
```

### CPU — H5 quantizado Vitis AI

Arquivos:

```text
results/vitisai/cpu_model_only.json
results/vitisai/cpu_host_to_host.json
```

| Modo | FPS médio | Latência média |
|---|---:|---:|
| Model-only | 1.647,85 FPS | 0,6077 ms |
| Host-to-host | 1.484,66 FPS | 0,6762 ms |

Esses resultados executam o modelo Keras quantizado através do TensorFlow na
CPU; não usam o XMODEL nem o DPU.

Comparando os números existentes, o model-only da ZCU104 apresentou cerca de
2,73 vezes o FPS do H5 quantizado na CPU. Já o host-to-host ficou próximo porque
o pequeno modelo deixa o custo do processamento Python/NumPy mais visível.

Essa comparação deve ser tratada como indicativa: os equipamentos, runtimes e
implementações são diferentes. Para uma comparação experimental rigorosa,
registre CPU, GPU e DPU com mesma carga do sistema, política de frequência,
temperatura, duração e definição de escopo.

## 11. Estrutura do JSON da ZCU104

```text
python
model
├── input_name / input_shape / input_fix_point
└── output_name / output_shape / output_fix_point
accuracy                         # somente quando --accuracy é usado
├── images / correct / accuracy
└── elapsed_seconds / images_per_second
model_only                       # somente quando --benchmark é usado
├── configuração e resumo
└── runs_data[]
    └── latências e FPS de cada run
host_to_host                     # somente quando --benchmark é usado
├── configuração e resumo
└── runs_data[]
    └── latências e FPS de cada run
single_image                     # somente quando --image é usado
└── path / prediction
```

`accuracy` é armazenada entre 0 e 1. Por exemplo, `0.95` significa 95%.

## 12. Copiar resultados de volta

No computador host:

```bash
scp root@IP_DA_ZCU104:/home/root/deploy/zcu104/results.json \
  results/zcu104/results.json
```

Se a pasta na placa estiver em outro local, use `pwd` dentro dela para descobrir
o caminho correto.

## 13. Recriar o pacote depois de alterações

No diretório raiz do projeto:

```bash
./scripts/package_zcu104.sh
```

Depois copie novamente o `.tar.gz` para a placa.

## 14. Diagnóstico de problemas

### `No module named vart` ou `No module named xir`

O script está fora da imagem Linux/runtime Vitis AI da placa. Confirme que a
imagem da ZCU104 possui VART/XIR para Vitis AI 3.5.

### `No module named cv2`

O OpenCV Python não está instalado na imagem. Use a imagem oficial compatível
ou instale o pacote apropriado para a distribuição da placa.

### Erro de fingerprint/DPU incompatível

O XMODEL foi compilado para outro DPU. Execute:

```bash
xdputil query
xdputil xmodel model/trafficsignnet_int8.xmodel -l
```

Se a placa usa bitstream customizado, recompile com o `arch.json` produzido por
esse hardware.

### Zero ou mais de um subgrafo DPU

O programa espera exatamente um subgrafo DPU, pois a rede compilada atual tem
um único kernel. Revise o relatório do compilador e operadores enviados para a
CPU.

### Acurácia incorreta

Confira, nesta ordem:

1. BGR foi convertido para RGB;
2. imagem foi redimensionada para 32x32;
3. normalização usa `/255.0`;
4. `fix_point` veio do tensor XMODEL;
5. nomes das pastas correspondem aos IDs das classes;
6. o XMODEL foi gerado a partir do H5 INT8 correto.

### Resultado muda entre execuções

Variações pequenas são normais. Carga do sistema, frequência do processador,
temperatura, serviços em segundo plano e throttling afetam principalmente os
percentis de cauda. Use warmup, runs longos e mantenha a placa nas mesmas
condições ao comparar versões.

## 15. Arquivos gerados automaticamente

`__pycache__/` e arquivos `.pyc` são caches do Python. Não fazem parte do código
fonte e são excluídos do pacote `.tar.gz`. `results.json` é sobrescrito quando
se usa o caminho padrão; use `--results` com nomes diferentes para preservar
várias medições.
