# Sobre este trabalho

Com os avanços tecnológicos na área da medicina veterinária, emergiu a oportunidade de desenvolver aplicações móveis destinadas a otimizar o processo de monitorização de animais. No seguimento do desenvolvimento da aplicação iEquus num Trabalho Final de Curso anterior, identificou-se a necessidade de integrar uma funcionalidade que, a partir de uma imagem de um membro de um cavalo, fosse capaz de entregar a correspondente imagem radiográfica do mesmo membro, mantendo o mesmo ângulo de captação. O presente trabalho propõe uma *pipeline* de visão computacional para a classificação de imagens de membros de cavalos, recorrendo ao *fine-tuning* de modelos de *deep learning* pré-treinados disponíveis na biblioteca PyTorch. Foram testadas 17 arquiteturas, e a seleção do modelo mais adequado baseou-se nas métricas de *accuracy*, *precision*, *recall*, *F1-score* e na matriz de confusão.

# Instalação de dependências

Uma vez descarregados os ficheiros do projeto, antes de executar qualquer código, devem ser seguidos três procedimentos.

Em primeiro lugar, a instalação dos pacotes e dependências, caso o ambiente de desenvolvimento (IDE) não o faça automaticamente, através do ficheiro requirements.txt, utilizando no terminal:

```bash
pip install -r requirements.txt
```

Em segundo lugar, a criação de uma conta na plataforma Hugging Face, caso ainda não possua uma, bem como o pedido de acesso ao modelo DINO-ViT à Meta e a geração de um *token* de acesso.

É importante associar o *token* ao ambiente de desenvolvimento, caso ainda não o tenha feito; caso contrário, não será possível aceder ao modelo DINO-ViT.

Em terceiro lugar, será necessária a obtenção dos modelos de classificação YOLO26 -m e -n no website da Ultralytics.

# Documentos

## Documentos executáveis

Uma vez instaladas as dependências, o código já pode ser executado. Existem quatro ficheiros principais com os quais o utilizador pode interagir:

* analiseexploratoria.ipynb
* TFC_codigo.ipynb
* melhores_modelos.py
* Resultados.ipynb

### requirements.txt

Este ficheiro contém todos os pacotes instalados no ambiente de trabalho no qual o projeto foi criado.

### analiseexploratoria.ipynb

Este notebook realiza uma análise exploratória dos dados, incluindo:

* Distribuição das classes
* Visualização e análise geral dos dados
* Correção dos dados
* Separação dos dados em treino/teste

**Requisitos:**

Os dados devem estar na pasta:

* Dados

### melhores_modelos.py

Script principal para treino dos modelos.

Contém a função run, que:

* Recebe parâmetros de configuração do treino
* Executa múltiplos treinos
* Avalia os modelos com base no F1-score

Permite treinar com:

* Dados reais
* Data augmentation simples
* Data augmentation personalizada

### TFC_codigo.ipynb

Notebook com:

* Tabelas de resultados
* Comparação entre modelos
* Análise dos efeitos de data augmentation

### Resultados.ipynb

Notebook com:

* Tabela com os resultados dos 7 melhores modelos
* Testes com o modelo ótimo

Para executar os ficheiros TFC_codigo.ipynb e Resultados.ipynb será necessário executar o ficheiro melhores_modelos.py, caso ainda não existam os ficheiros CSV necessários.

## Documentos não executáveis

### utils.py

Este documento contém todas as funções auxiliares utilizadas ao longo de todo o TFC.


## Organização dos dados

Os dados utilizados no projeto encontram-se divididos em diferentes diretórios, de acordo com a fase de desenvolvimento.

Os resultados obtidos na entrega anterior estão armazenados nas pastas `aug_f1-4` e `treino_modelos`, que contêm os experimentos realizados com diferentes abordagens de data augmentation e treino de modelos.

Nesta última entrega, os dados mais recentes encontram-se organizados na pasta `runs`, a qual contém os resultados associados ao modelo YOLO26, e no ficheiro `teste`, que inclui os dados utilizados para avaliação final do modelo.
