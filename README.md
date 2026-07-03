# 🦾 Sistema Inteligente para Reconhecimento de Objetos de EPI Usando Visão Computacional

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Sistema inteligente de visão computacional desenvolvido para reconhecer o uso de **Equipamentos de Proteção Individual (EPIs)** em trabalhadores, com foco em **capacetes**, **óculos de proteção** e **máscaras** no contexto da **construção civil**.

O projeto foi desenvolvido no âmbito do **Programa de Iniciação Científica da UTFPR**, com bolsa **CNPq**, e compara arquiteturas de **Redes Neurais Artificiais (ANNs)** e **Redes Neurais Convolucionais (CNNs)** para classificação de imagens relacionadas ao uso de EPIs.

---

## 📌 Sobre o projeto

A segurança no ambiente de trabalho é essencial em setores de risco. Na construção civil, o uso correto de EPIs pode reduzir acidentes e auxiliar na fiscalização das condições de segurança.

Este projeto propõe uma solução baseada em **redes neurais** e **visão computacional** para classificar imagens e identificar automaticamente a presença de EPIs. Foram comparadas duas abordagens principais:

- **ANN – Artificial Neural Network**, ou Rede Neural Artificial tradicional;
- **CNN – Convolutional Neural Network**, ou Rede Neural Convolucional.

A pesquisa avaliou diferentes topologias para cada arquitetura, analisando curvas de acurácia, curvas de perda, matrizes de confusão, tabelas comparativas e testes práticos com imagens reais.

---

## 🎯 Objetivos

- Desenvolver um sistema inteligente para reconhecimento de EPIs em imagens.
- Identificar itens como capacete, óculos de proteção, máscara e combinações entre eles.
- Comparar o desempenho entre modelos ANN e CNN.
- Avaliar a capacidade de generalização dos modelos em testes reais.
- Investigar limitações das arquiteturas clássicas e apontar melhorias futuras com modelos de detecção em tempo real, como YOLOv9.

---

## 🧪 Metodologia

Foram utilizadas **360 imagens**, igualmente distribuídas entre as classes de interesse, incluindo:

- óculos;
- capacete;
- máscara;
- combinações entre os EPIs;
- ausência de equipamentos.

As imagens foram pré-processadas e normalizadas para o intervalo **[0, 1]**. O relatório descreve o redimensionamento para **224 × 224 px**; porém, os scripts atuais do repositório utilizam **64 × 64 px**, mantendo compatibilidade com a implementação salva no GitHub.

Os modelos foram treinados por **70 épocas**, com acompanhamento de:

- `accuracy` de treino e validação;
- `loss` de treino e validação;
- matrizes de confusão;
- comparação entre topologias;
- testes práticos com imagens reais.

---

## 🧠 Arquiteturas avaliadas

Foram testadas cinco configurações para cada família de rede: um modelo base e quatro variações.

### Variações da ANN

| Topologia | Descrição |
|---|---|
| Modelo Base Inicial | Arquitetura inicial de referência |
| ReduzComplexidade | Menor quantidade de neurônios por camada |
| RedeMaisProfunda | Mais camadas ocultas |
| BatchNormalization | Normalização para estabilizar o treinamento |
| FunilInvertido | Organização das camadas em formato de funil invertido |

### Variações da CNN

| Topologia | Descrição |
|---|---|
| Modelo Base Inicial | Arquitetura convolucional inicial |
| BatchNormalization | Normalização entre camadas |
| AumentaProfundidade | Rede mais profunda para melhorar a extração de padrões |
| ArquiteturaRasa | Menor número de camadas convolucionais |
| GlobalAveragePooling | Substituição do `Flatten` por `GlobalAveragePooling2D` para reduzir parâmetros |

---

## 📊 Resultados

### Resultados das ANNs

| Topologia | Loss Final | Acurácia Final |
|---|---:|---:|
| Modelo Base Inicial | 0,3538 | 55,60% |
| ReduzComplexidade | 0,9752 | 62,50% |
| RedeMaisProfunda | 1,4956 | 33,33% |
| BatchNormalization | 1,0717 | 73,61% |
| FunilInvertido | 2,0174 | 29,17% |

A melhor configuração entre as ANNs foi a arquitetura com **BatchNormalization**, alcançando **73,61% de acurácia final**.

### Resultados das CNNs

| Topologia | Loss Final | Acurácia Final |
|---|---:|---:|
| Modelo Base Inicial | 0,3538 | 93,06% |
| BatchNormalization | 0,2796 | 91,67% |
| AumentaProfundidade | 0,1164 | 97,22% |
| ArquiteturaRasa | 0,2685 | 94,44% |
| GlobalAveragePooling | 0,2462 | 93,06% |

A melhor configuração geral foi a **CNN AumentaProfundidade**, que atingiu **97,22% de acurácia** e **loss final de 0,1164**.

---

## 🔎 Discussão dos resultados

Os resultados mostraram que as **CNNs tiveram desempenho superior às ANNs**, principalmente por serem mais adequadas ao processamento de imagens, já que conseguem extrair padrões visuais como formas, bordas e texturas.

Apesar da alta acurácia da CNN em validação, os testes práticos indicaram limitações na identificação de **óculos de proteção**, especialmente quando apareciam isolados ou combinados com outros EPIs. Esse comportamento reforçou a necessidade de explorar arquiteturas mais robustas para detecção de objetos.

Como melhoria futura, o uso de **YOLOv9** foi apontado como alternativa promissora, por permitir a identificação e localização de múltiplos objetos em tempo real em uma mesma imagem.

---

## 🧰 Tecnologias utilizadas

- **Python 3.10+**
- **TensorFlow / Keras**
- **OpenCV**
- **Matplotlib**
- **Seaborn**
- **NumPy**
- **Pandas**
- **Scikit-learn**

---

## 📁 Estrutura do repositório

```text
.
├── data/
│   └── README.md                         # Descrição do dataset utilizado
│
├── docs/
│   ├── relatorio_final.docx              # Relatório final em Word
│   └── relatorio_final.pdf               # Relatório final em PDF
│
├── models/
│   └── README.md                         # Pasta opcional para modelos treinados
│
├── results/
│   ├── figures/                          # Gráficos de acurácia/loss e imagens de exemplo
│   │   ├── ann_base_accuracy.png
│   │   ├── ann_base_loss.png
│   │   ├── cnn_aumenta_profundidade_accuracy.png
│   │   └── ...
│   │
│   ├── confusion_matrices/               # Matrizes de confusão das variações
│   │   ├── ann_base_confusion_matrix.png
│   │   ├── cnn_base_confusion_matrix.png
│   │   └── ...
│   │
│   ├── tables/                           # Tabelas do relatório em CSV
│   │   ├── resultados_ann.csv
│   │   ├── resultados_cnn.csv
│   │   ├── teste_real_ann.csv
│   │   └── teste_real_cnn.csv
│   │
│   └── mapeamento_figuras.csv            # Relação entre figuras do relatório e arquivos PNG
│
├── src/
│   ├── ann_model.py                      # Script principal da ANN atualmente salvo
│   ├── cnn_model.py                      # Script principal da CNN atualmente salvo
│   ├── ann_variations.py                 # Funções com as variações ANN
│   ├── cnn_variations.py                 # Funções com as variações CNN
│   ├── train_all_variations.py           # Script para treinar todas as variações
│   └── README_variacoes_modelos.md       # Instruções específicas das variações
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📂 Observação sobre os dados

O dataset completo não está incluído neste repositório por questões de privacidade e tamanho dos arquivos. A pasta `data/` contém uma descrição da organização utilizada durante os experimentos.

Caso deseje reproduzir os treinamentos, organize as imagens localmente e gere um arquivo CSV no formato:

```csv
image_path,label
captured_data/exemplo_001.png,1
captured_data/exemplo_002.png,3
```

O script `src/train_all_variations.py` espera, por padrão, o arquivo:

```text
captured_data/mask_data.csv
```

---

## 🚀 Como executar

Clone o repositório:

```bash
git clone https://github.com/marianagpalacios/Sistema-De-Visao-Computacional-Para-Reconhecimento-Inteligente-De-Objetos.git
cd Sistema-De-Visao-Computacional-Para-Reconhecimento-Inteligente-De-Objetos
```

Crie e ative um ambiente virtual:

```bash
python -m venv venv
```

No Linux/Mac:

```bash
source venv/bin/activate
```

No Windows:

```bash
venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### Executar os scripts principais

```bash
# Rede Neural Artificial atualmente salva no projeto
python src/ann_model.py

# Rede Neural Convolucional atualmente salva no projeto
python src/cnn_model.py
```

### Treinar todas as variações

```bash
python src/train_all_variations.py --family all
```

### Treinar somente as ANNs

```bash
python src/train_all_variations.py --family ann
```

### Treinar somente as CNNs

```bash
python src/train_all_variations.py --family cnn
```

### Treinar uma variação específica

```bash
python src/train_all_variations.py --only ann_batchnormalization
python src/train_all_variations.py --only cnn_aumenta_profundidade
```

### Alterar o tamanho das imagens

Por padrão, o script usa **64 × 64 px**, que é o tamanho utilizado nos scripts atuais:

```bash
python src/train_all_variations.py --family all --image-size 64
```

Para testar com **224 × 224 px**, conforme descrito no relatório:

```bash
python src/train_all_variations.py --family all --image-size 224
```

### Salvar os modelos treinados

```bash
python src/train_all_variations.py --family all --save-models
```

Os modelos serão salvos em:

```text
models/
```

---

## ✅ Verificação dos arquivos atuais

Os scripts de variações foram adicionados para documentar as arquiteturas testadas no relatório. A identificação dos scripts principais ficou assim:

| Arquivo | Variação identificada |
|---|---|
| `src/ann_model.py` | ANN com BatchNormalization |
| `src/cnn_model.py` | CNN AumentaProfundidade, com BatchNormalization e GlobalAveragePooling2D |

---

## ✅ Conclusão

A pesquisa confirmou a viabilidade do uso de visão computacional e redes neurais para reconhecimento automatizado de EPIs. As **CNNs**, especialmente a topologia **AumentaProfundidade**, apresentaram o melhor equilíbrio entre precisão, robustez e capacidade de generalização.

As **ANNs** também se mostraram úteis em cenários mais simples ou com menor custo computacional, mas apresentaram desempenho inferior em comparação às redes convolucionais.

Os testes práticos evidenciaram limitações na detecção de óculos, indicando que modelos de detecção em tempo real, como o **YOLOv9**, podem ampliar a aplicabilidade do sistema em ambientes reais de trabalho.

---

## 👩‍💻 Autores

- **Mariana Gasparotto Palácios** – Bolsista CNPq – Engenharia de Software – UTFPR
- **Prof. Dr. Márcio Mendonça** – Orientador

---

## 🏛️ Instituição

**Universidade Tecnológica Federal do Paraná – UTFPR**  
Campus Cornélio Procópio  
Área do conhecimento: Sistemas de Computação

---

## 📄 Licença

Este projeto está licenciado sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais informações.
