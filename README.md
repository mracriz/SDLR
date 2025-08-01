# SDLR: Destilação de Conhecimento para Learning to Rank 🚀

Este repositório contém uma implementação da técnica de Destilação de Conhecimento (*Knowledge Distillation*) aplicada a modelos de Learning to Rank, baseada no framework `allRank`. O fluxo principal, chamado SDLR, envolve o treinamento de um modelo "Professor" (Teacher) para guiar o aprendizado de um modelo "Aluno" (Student) menor e mais eficiente.

O projeto utiliza o **MLflow** para rastrear, comparar e gerenciar todos os experimentos de forma robusta e organizada.

## Origem e Agradecimentos

Esta é uma implementação e adaptação do trabalho original sobre SDLR dos seguintes autores. Recomenda-se a leitura do trabalho original para um entendimento completo da metodologia.

  * **Repositório Original SDLR:** [https://github.com/sanazkeshvari/Papers/tree/main/SDLR](https://github.com/sanazkeshvari/Papers/tree/main/SDLR)

Ambos os projetos são construídos sobre o excelente framework de Learning to Rank `allRank`.

  * **allRank Framework:** [https://github.com/allegro/allRank](https://github.com/allegro/allRank)

## 🛠️ Pré-requisitos e Instalação

Antes de começar, certifique-se de que você tem **Python 3.10** (ou superior) instalado.

### Passo 1: Crie e Ative o Ambiente Virtual (`venv`)

É altamente recomendado usar um ambiente virtual para isolar as dependências do projeto.

```bash
# Na pasta raiz do projeto (SDLR), crie o ambiente virtual
python3 -m venv venv

# Ative o ambiente (no macOS/Linux)
source venv/bin/activate
```

> *No Windows, o comando para ativar é `venv\Scripts\activate`*

### Passo 2: Instale as Dependências

A instalação tem duas partes: primeiro, instalamos os pacotes padrão do `requirements.txt` da raiz. Depois, "linkamos" os projetos locais do Teacher e Student ao nosso ambiente.

```bash
# 1. Instale as bibliotecas principais a partir do arquivo da raiz
pip install -r requirements.txt

# 2. "Linke" o código do Teacher para que o Python o encontre
pip install -e Teacher/allRank-master/

# 3. "Linke" o código do Student para que o Python o encontre
pip install -e Student/allRank-master/
```

> **Nota Importante ⚠️:** Os comandos `pip install -e` são **essenciais**. Eles criam um link para seus projetos locais, resolvendo os erros de `ModuleNotFoundError` e `DistributionNotFound` e garantindo que o Python e o MLflow consigam encontrar seus scripts de treino.

## 💾 Estrutura e Preparação dos Dados

O framework espera que os dados de treino, validação e teste estejam em uma estrutura de pastas específica e no formato correto.

### Formato dos Arquivos

Todos os seus arquivos de dados (`train.txt`, `vali.txt`, `test.txt`) **PRECISAM ESTAR NO FORMATO SVM Rank**. Você pode usar o script `utils/csv_to_svm.py` incluído neste repositório para converter seus dados de `.csv` para este formato.

### Estrutura de Pastas

Dentro da pasta que você especificar no `config.json` (no campo `data.path`), o framework espera encontrar os seguintes arquivos, **exatamente com estes nomes**:

```
/seu/caminho/para/os/dados/
├── train.txt
├── vali.txt
└── test.txt
```

## ⚙️ Configuração dos Experimentos

A configuração de cada modelo é controlada por um arquivo `config.json`. **É obrigatório editar estes arquivos antes de executar um treino.**

  * **Config do Teacher:** `Teacher/allRank-master/allrank/in/config.json`
  * **Config do Student:** `Student/allRank-master/allrank/in/config.json`

O parâmetro mais importante a ser editado em ambos os arquivos é o `data.path`, que deve apontar para o diretório que contém os arquivos `train.txt`, `vali.txt` e `test.txt`.

```json
{
  "data": {
    "path": "/caminho/absoluto/para/sua/base_de_dados",
    "validation_ds_role": "vali"
  }
}
```

> **Nota:** Certifique-se de que `validation_ds_role` está definido como `"vali"` para que o arquivo `vali.txt` seja usado para validação durante o treino. O arquivo `test.txt` será usado implicitamente como conjunto de teste pelo data loader.

## ⚡ Como Executar os Experimentos

Existem dois scripts orquestradores na raiz do projeto (`SDLR/`).

-----

### 1\. Fluxo de Destilação SDLR (Teacher -\> Student) 👨‍🏫 -\> 👨‍🎓

Para executar o fluxo completo de destilação de conhecimento, use o script `run_experiment.py`.

**Comando de Exemplo:**

```bash
python3 run_experiment.py "Meu_Experimento_SDLR" \
  --inference_data "/caminho/absoluto/para/sua/base_de_teste_final.txt"
```

  * **`--inference_data`**: Este é o parâmetro para a sua base de teste final, que também **PRECISA ESTAR NO FORMATO SVM Rank**.

-----

### 2\. Treino de um Modelo Único (Baselines) 🔬

Para treinar um único modelo (como `NeuralNDCG`) como base de comparação, use `run_single_model_experiment.py`.

**Comando de Exemplo:**

```bash
python3 run_single_model_experiment.py "Meu_Experimento_NeuralNDCG" \
  "Teacher/allRank-master/allrank/in/config_neuralNDCG.json" \
  --inference_data "/caminho/absoluto/para/sua/base_de_teste_final.txt"
```

  * **`--inference_data`**: A sua base de teste final, em **FORMATO SVM Rank**.

## 📊 Visualizando os Resultados com MLflow

Para ver os resultados de forma organizada:

1.  **Inicie a UI do MLflow:** No terminal, a partir da pasta raiz `SDLR`, execute:

    ```bash
    mlflow ui
    ```

2.  **Acesse no Navegador:** Abra seu navegador e acesse `http://127.0.0.1:5000`.

Lá, você encontrará todos os seus experimentos, poderá comparar as métricas (`final_ndcg_at_k`), visualizar os parâmetros e explorar os artefatos de cada execução. Divirta-se analisando\!
