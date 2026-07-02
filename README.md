# 🦾 Sistema Inteligente para Reconhecimento de Objetos de EPI Usando Visão Computacional

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Sistema inteligente de visão computacional desenvolvido para reconhecer o uso de **Equipamentos de Proteção Individual (EPIs)** em trabalhadores, com foco em itens como **capacetes**, **óculos de proteção** e **máscaras** no contexto da **construção civil**.

O projeto foi desenvolvido no âmbito do **Programa de Iniciação Científica da UTFPR**, com bolsa **CNPq**, e apresentado no **SICITE – Seminário de Iniciação Científica e Tecnológica**, evento dedicado à divulgação de pesquisas desenvolvidas em diversas áreas da ciência.

---

## 📌 Sobre o projeto

A segurança no ambiente de trabalho é essencial em setores de risco. Na construção civil, o uso correto de EPIs pode reduzir acidentes e auxiliar na fiscalização das condições de segurança.

Este projeto propõe uma solução baseada em **redes neurais** e **visão computacional** para classificar imagens e identificar automaticamente a presença de EPIs. Foram comparadas duas abordagens principais:

- **ANN – Artificial Neural Network**, ou Rede Neural Artificial tradicional;
- **CNN – Convolutional Neural Network**, ou Rede Neural Convolucional.

A pesquisa avaliou diferentes topologias para cada arquitetura, analisando desempenho, perda, acurácia, matrizes de confusão e testes práticos com imagens reais.

---

## 🎯 Objetivos

- Desenvolver um sistema inteligente para reconhecimento de EPIs em imagens.
- Identificar itens como capacete, óculos de proteção, máscara e combinações entre eles.
- Comparar o desempenho entre modelos ANN e CNN.
- Avaliar a capacidade de generalização dos modelos em testes reais.
- Investigar limitações das arquiteturas clássicas e apontar melhorias futuras com modelos de detecção em tempo real, como YOLOv9.

---

## 🧪 Metodologia

### 1. Coleta e preparação dos dados

Foram utilizadas **360 imagens**, igualmente distribuídas entre as classes de interesse, incluindo:

- óculos;
- capacete;
- máscara;
- combinações entre os EPIs;
- ausência de equipamentos.

As imagens passaram por pré-processamento com:

- redimensionamento para **224 × 224 px**;
- normalização dos pixels para o intervalo **[0, 1]**;
- organização das classes para treinamento, validação e testes.

### 2. Treinamento dos modelos

Os modelos foram treinados por **70 épocas**, com acompanhamento de:

- `accuracy` de treino e validação;
- `loss` de treino e validação;
- matrizes de confusão;
- comparação entre topologias.

Foram testadas cinco configurações para cada família de rede.

#### Variações da ANN

| Topologia | Descrição |
|---|---|
| Modelo Base Inicial | Arquitetura inicial de referência |
| ReduzComplexidade | Menor quantidade de neurônios |
| RedeMaisProfunda | Mais camadas ocultas |
| BatchNormalization | Normalização para estabilizar o treinamento |
| FunilInvertido | Organização das camadas em formato invertido |

#### Variações da CNN

| Topologia | Descrição |
|---|---|
| Modelo Base Inicial | Arquitetura convolucional inicial |
| BatchNormalization | Normalização entre camadas |
| AumentaProfundidade | Rede mais profunda para melhor extração de padrões |
| ArquiteturaRasa | Menos camadas convolucionais |
| GlobalAveragePooling | Redução de overfitting e simplificação da rede |

### 3. Avaliação

A avaliação considerou:

- curvas de acurácia;
- curvas de loss;
- matrizes de confusão;
- testes práticos com imagens reais;
- comparação entre desempenho de ANN e CNN.

Nos testes reais, foram utilizadas **10 amostras por classe**, totalizando **90 imagens por modelo**.

### Observação sobre os dados

O dataset completo não está incluído neste repositório por questões de privacidade e tamanho dos arquivos. A pasta `data/` contém uma descrição da organização utilizada durante os experimentos.

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
- **Jupyter Notebook**

---

## 📁 Estrutura do repositório

```text
.
├── results/              # Resultados, gráficos e matrizes de confusão
├── src/                  # Códigos-fonte dos modelos ANN e CNN
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
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

Execute os modelos:

```bash
# Rede Neural Artificial
python src/ANNcorrigido.py

# Rede Neural Convolucional
python src/CNNcorrigido.py
```

> Observação: antes da execução, verifique se os caminhos do conjunto de imagens estão configurados corretamente nos arquivos da pasta `src/`.

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
