# Sobre este trabalho
Com os avanços tecnológicos na área da medicina veterinária, emergiu a oportunidade de desenvolver aplicações móveis destinadas a otimizar o processo de monitorização de animais. No seguimento do desenvolvimento da aplicação iEquus num Trabalho Final de Curso anterior, identificou-se a necessidade de integrar uma funcionalidade que, a partir de uma imagem de um membro de um cavalo, fosse capaz de entregar a correspondente imagem radiográfica do mesmo membro, mantendo o mesmo ângulo de captação. O presente trabalho propõe uma _pipeline_ de visão computacional para a classificação de imagens de membros de cavalos, recorrendo ao _fine-tuning_ de modelos de _deep learning_ pré-treinados disponíveis na biblioteca PyTorch. Foram testadas 15 arquiteturas, e a seleção do modelo mais adequado baseou-se nas métricas de _accuracy_, _precision_, _recall_, _F1-score_ e a matriz de confusão.

# Instalação de dependências
Uma vez descarregados os arquivos do projeto, antes de executar qualquer código, devem ser seguidos dois procedimentos. 
Em primeiro lugar, a instalação dos pacotes e dependências, caso o ambiente de desenvolvimento (IDE) não o faça automaticamente, através do arquivo requirements.txt, utilizando no terminal 
```
pip install -r requirements.txt
```
Em segundo lugar, a criação de uma conta na plataforma Hugging Face, caso ainda não possua uma, bem como o pedido de acesso ao modelo DINO-ViT à Meta e a geração de um _token_ de acesso.

É importante associar o _token_ ao ambiente de desenvolvimento, caso ainda não o tenha feito; caso contrário, não será possível aceder ao modelo DINO-ViT.

# Documentos

## Documentos executáveis
Uma vez as dependências instaladas, o código já pode ser executado. Existem três ficheiros principais com os quais o utilizador pode interagir: 
- analiseexploratoria.ipynb, 
- TFC_codigo.ipynb
- melhores_modelos.py


### requirements.txt
Este txt contém todos os pacotes instalados no ambiente de trabalho na qual o projeto foi criado.


### analiseexploratoria.ipynb
Este notebook realiza uma análise exploratória dos dados, incluindo:
- Distribuição das classes
- Visualização e análise geral dos dados
- Correção dos dados
- Separação dos dados treino/teste
**Requisitos:**
Os dados devem estar nas pastas:
- Dados


### melhores_modelos.py
Script principal para treino dos modelos.
Contém a função run, que:
- Recebe parâmetros de configuração do treino
- Executa múltiplos treinos
- Avalia os modelos com base no F1-score
Permite treinar com:
- Dados reais
- Data augmentation simples
- Data augmentation personalizada

### TFC_codigo.ipynb
Notebook com:
- Tabelas de resultados
- Comparação entre modelos
- Análise dos efeitos de data augmentation
Para executar o ficheiro TFC_codigo.ipynb será necessário rodar o arquivo melhores_modelos.py caso ainda não existam os ficheiros CSV necessários.


## Documentos não executáveis

### utils.py
Este documento contém todos as funções auxiliares utilizadas ao longo de todo o TFC

###  split_data:
Esta função divide os dados em 80/20 treino e teste e guarda tudo em uma nova pasta "dados_split"

### img_process:
Aplica transformações aos dados:
- train_transforms
- test_transforms
Cria batches de tamanho 16.
**Requisitos:**
Os dados devem estar nas pastas:
- dados_split

### best_epoch
Esta função treina e testa um modelo a cada epoch para achar a epoch com o melhor valor de f1.
**Parâmetros:**
- path = nome da pasta onde este irá guardar os modelos
- nome = nome do modelo
- model = o modelo em si
- optimizer = o nome do otimizador
- loss_type = tipo de função de perda
- train_loader = os dados de treino após sofrerem com os métodos de transformação de dados.
- test_loader = os dados de teste após sofrerem com os métodos de transformação de dados.
- num_epochs = numero de epochs total que o modelo deve treinar
**Esta retorna:**
- best_prev = lista das classes previstas pelo modelo
- best_real = lista das classes reais das imagens
- best_epoch_num = O numero da epoch na qual teve o melhor resultado de f1 
- historico_f1 = Lista com os valores de f1-macro de todas as epochs

### metricas
Esta função:
- calcula métricas de accuracy, precision, recall, f1
- Cria gráfico com os valores de f1/epoch
- Cria uma matriz de confusão
- Atualiza um dataframe com os valores das métricas calculadas
**Parâmetros:**
- path = nome da pasta onde este irá guardar os modelos
- nome = nome do modelo
- best_prev = lista das classes previstas pelo modelo
- best_real = lista das classes reais das imagens
- transform = a transformação de imagem aplicada aos dados de teste.
- df_resultados = dataframe que contem os resultados do modelo e dos modelos em teste.
- best_epoch_num = O numero da epoch na qual teve o melhor resultado de f1 
- historico_f1 = Lista com os valores de f1-macro de todas as epochs

### modelos
Função que cria um modelo que será posteriormente treinado
**Parâmetros:**
- model_name = nome do modelo
- learning_rate = O learning rate na qual a função terá que treinar com
- optim = O otimizador da função.
**Este retorna:**
- model = A arquitetura do modelo
- optimizer = O otimizador


### get_head_params
Função que dependendo do modelo indicado, retorna a camada final do modelo
**Parâmetros:**
- model = A arquitetura do modelo
- model_name = O nome do modelo


### train_and_evaluate_models
Função que concatena todas as funções a cima em uma só função
Esta função cria a arquitetura, treina a mesma e depois calcula suas métricas
Cria um CSV com as métricas do modelo treinado e os modelos anteriormente treinados
**Parâmetros:**
- path = nome da pasta onde este irá guardar os modelos
- file_path = O nome do arquivo csv que será criado
- modelos_config = Um dicionário que contem os valores do numero máximo de epochs de treino, learning rate, optmizador, e função de perda.
- train_transforms = método de transformação de dado para os dados de treino
- test_transforms = método de transformação de dado para os dados de teste
- df_resultados = dataframe que contem os resultados do modelo e dos modelos em teste.


### load_or_create
Função que retorna um arquivo CSV, se não houver, cria e retorna um dataframe


### run_experiments
Função que define uma seed e depois treina os modelos


