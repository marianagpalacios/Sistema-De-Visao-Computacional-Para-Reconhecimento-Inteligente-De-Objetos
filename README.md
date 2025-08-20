# 🦾 Sistema de Visão Computacional para Reconhecimento de EPIs

Este projeto tem como objetivo o desenvolvimento de um **sistema inteligente de visão computacional** para reconhecimento de **Equipamentos de Proteção Individual (EPIs)**, como **óculos de proteção, capacetes e máscaras**, aplicados ao contexto da **construção civil**.  

A pesquisa foi desenvolvida no âmbito do **Programa de Iniciação Científica (UTFPR - CNPq)**.

---

## 📌 Objetivos

- Detectar automaticamente o uso de EPIs em trabalhadores.  
- Comparar diferentes arquiteturas de redes neurais:  
  - **Redes Neurais Artificiais (ANNs)**  
  - **Redes Neurais Convolucionais (CNNs)**  
- Explorar arquiteturas de **detecção em tempo real** com **YOLOv9**.  
- Avaliar e comparar o desempenho de cada abordagem.  

---

## 🧪 Metodologia

1. **Coleta e pré-processamento de dados**  
   - 360 imagens divididas entre as classes de interesse (óculos, capacete, máscara e combinações).  
   - Redimensionamento para **224×224 px** e normalização dos pixels em `[0,1]`.  

2. **Treinamento dos modelos**  
   - ANN e CNN treinadas por **70 épocas**.  
   - Variações de topologia para cada arquitetura (BatchNormalization, Aumento de profundidade, etc.).  

3. **Avaliação de desempenho**  
   - Curvas de *accuracy* e *loss*.  
   - Matrizes de confusão.  
   - Testes práticos com imagens reais.  

4. **Integração do YOLOv9**  
   - Implementação para detecção em tempo real.  
   - Capacidade de identificar múltiplos objetos em um único quadro.  

---

## 📊 Resultados principais

- **CNNs** apresentaram desempenho superior às ANNs, alcançando até **97,22% de acurácia** na validação.  
- A **ANN com BatchNormalization** teve bom equilíbrio, mas menor desempenho geral.  
- O **YOLOv9** superou as limitações, oferecendo **detecção robusta, rápida e em tempo real**.  

---

## 📚 Tecnologias utilizadas

- **Python 3.10+**  
- **[TensorFlow](https://www.tensorflow.org/)** / **[Keras](https://keras.io/)** – Modelos ANN e CNN  
- **[PyTorch](https://pytorch.org/)** – Implementação do YOLOv9  
- **[OpenCV](https://opencv.org/)** – Processamento de imagens  
- **[Matplotlib](https://matplotlib.org/)** / **[Seaborn](https://seaborn.pydata.org/)** – Visualização de resultados  
- **[Jupyter Notebook](https://jupyter.org/)** – Experimentação e análises  

---

## 👩‍💻 Autores

- **Mariana Gasparotto Palácios** – Bolsista CNPq – Engenharia de Software  
- **Prof. Dr. Márcio Mendonça** – Orientador  

---

## 🚀 Como executar

Clone o repositório:
```bash
git clone https://github.com/usuario/Sistema-De-Visao-Computacional-Para-Reconhecimento-Inteligente-De-Objetos.git
cd Sistema-De-Visao-Computacional-Para-Reconhecimento-Inteligente-De-Objetos

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

# Rede Neural Artificial (ANN)
python src/ANNcorrigido.py

# Rede Neural Convolucional (CNN)
python src/CNNcorrigido.py






