# Titanic com SOLID na AWS
## 1. Objetivo

<p align="justify">Pipeline de Machine Learning robusto para o conjunto de dados Titanic, desenvolvido com princípios de Programação Orientada a Objetos (OOP), SOLID e padrões de projeto profissionais implementados por meio do AWS SageMaker Python SDK.</p>

---

## 2. Padrões Arquiteturais e de Projeto

<p align="justify">O projeto é modularizado em cinco componentes distintos para garantir responsabilidade única, manutenibilidade e separação de preocupações:</p>

<p align="justify">1. Transform Pattern (<code>preprocessing.py</code>): Isola a engenharia de atributos, a imputação de valores ausentes (<code>SimpleImputer</code>) e o escalonamento (<code>StandardScaler</code>) dentro de uma classe transformadora reutilizável, serializando-a para evitar <em>Training-Serving Skew</em>.</p>

<p align="justify">2. Reproducible Model Pattern (<code>train.py</code>): Encapsula o treinamento do modelo usando <code>XGBoost</code>, com sementes aleatórias fixas e hash de configuração para resultados determinísticos e reprodutíveis.</p>

<p align="justify">3. Workflow Pipeline (<code>orchestrator.py</code>): Orquestra a execução de ponta a ponta, ligando as etapas de processamento, treinamento e registro do modelo usando a <code>PipelineSession</code> do SageMaker.</p>

<p align="justify">4. Explainable Predictions Pattern (<code>inference.py</code>): Implementa os ganchos de inferência do SageMaker (<code>model_fn</code>, <code>predict_fn</code>), incorporando métricas de explicabilidade em tempo real via <code>SHAP</code>.</p>

<p align="justify">5. Model Versioning (<code>orchestrator.py</code>): Registra os artefatos treinados diretamente no AWS SageMaker Model Registry, sob um grupo de pacotes de modelo.</p>

---

## 3. Estrutura do Projeto

```text
Titanic/
├── data/
│   └── titanic.csv         # Dataset bruto do Titanic
├── requirements.txt        # Dependências do Python
├── preprocessing.py        # Etapa 1: Transformação de dados e engenharia de atributos
├── train.py                # Etapa 2: Treinamento reprodutível do modelo
├── inference.py            # Etapa 4: Inferência em Endpoint e explicações SHAP
└── orchestrator.py         # Orquestração do pipeline de fluxo de trabalho
```

---

## 4. Pré-requisitos

<p align="justify">É necessária uma conta ativa da AWS com permissões para o SageMaker, S3 e IAM. Também é preciso ter a AWS CLI instalada e configurada por meio do comando <code>aws configure</code>, além do Python 3.8 ou superior instalado localmente.</p>

---

## 5. Guia Passo a Passo para Implantação

### 5.1. Instalar e configurar a AWS CLI

<p align="justify">No Ubuntu ou em outra distribuição Linux de arquitetura x86-64, instale os utilitários necessários, baixe o instalador oficial da AWS CLI v2 e execute a instalação:</p>

```bash
sudo apt update
sudo apt install -y curl unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

<p align="justify">Se estiver usando Linux ARM64, utilize o instalador correspondente à arquitetura ARM64. Para verificar se a instalação funcionou, execute:</p>

```bash
aws --version
```

<p align="justify">Em seguida, configure as credenciais e a região padrão da AWS:</p>

```bash
aws configure
```

<p align="justify">O comando solicitará quatro valores: <code>AWS Access Key ID</code>, <code>AWS Secret Access Key</code>, região padrão e formato de saída. Informe as credenciais de um usuário ou perfil do IAM autorizado, por exemplo, a região <code>us-east-1</code> e o formato <code>json</code>. Use a região em que pretende criar os recursos do SageMaker e do S3. Não compartilhe nem publique suas chaves de acesso.</p>

<p align="justify">Para confirmar que a autenticação funcionou e conferir a identidade configurada, execute:</p>

```bash
aws sts get-caller-identity
```

<p align="justify">Se o comando retornar o identificador da conta e o ARN da identidade, a configuração foi reconhecida. A AWS CLI precisa de credenciais válidas e permissões adequadas; o comando <code>aws configure</code> não concede permissões por si só.</p>

---

### 5.2. Clonar e navegar para o repositório

<p align="justify">Execute os comandos abaixo para clonar o repositório e entrar na pasta do projeto. Substitua o endereço pelo endereço real do seu repositório.</p>

```bash
git clone https://github.com/seu-usuario/Titanic.git
cd Titanic
```

### 5.3. Configurar o ambiente virtual Python

<p align="justify">Crie e ative um ambiente virtual e instale as dependências definidas no arquivo <code>requirements.txt</code>.</p>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.4. Descobrir ou criar a função de execução do SageMaker via CLI

<p align="justify">O script precisa de uma função do IAM com permissões para executar as operações do SageMaker.</p>

<p align="justify">Para listar as funções existentes na sua conta:</p>

```bash
aws iam list-roles --query "Roles[?contains(RoleName, 'SageMaker')].Arn" --output text
```

<p align="justify">Se aparecer uma função, por exemplo <code>arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-...</code>, copie o valor da ARN.</p>

<p align="justify">Se você não tiver uma função e quiser criar uma pela CLI, execute os comandos abaixo para criar a política de confiança, criar a função e anexar a política de permissões necessária.</p>

```bash
cat << EOF > trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role --role-name AmazonSageMaker-ExecutionRole-Titanic --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy --role-name AmazonSageMaker-ExecutionRole-Titanic --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

rm trust-policy.json
```

<p align="justify">Em seguida, obtenha a ARN da função recém-criada:</p>

```bash
aws iam get-role --role-name AmazonSageMaker-ExecutionRole-Titanic --query 'Role.Arn' --output text
```

### 5.5. Configurar variáveis de ambiente

<p align="justify">Defina a função do IAM como uma variável de ambiente no terminal, substituindo o exemplo pela ARN real obtida na etapa anterior.</p>

```bash
export SAGEMAKER_ROLE="arn:aws:iam::SEU_ACCOUNT_ID:role/AmazonSageMaker-ExecutionRole-Titanic"
```

### 5.6. Configurar ou verificar o bucket S3

<p align="justify">O pipeline utiliza o bucket padrão do SageMaker, no formato <code>sagemaker-regiao-id_da_conta</code>.</p>

<p align="justify">Para verificar um bucket existente com nome iniciado por <code>sagemaker-</code>:</p>

```bash
aws s3 ls s3://$(aws s3api list-buckets --query "Buckets[?starts_with(Name, 'sagemaker-')].Name | [0]" --output text)
```

<p align="justify">Se ocorrer um erro indicando que o bucket não existe, você poderá criá-lo pela AWS CLI, substituindo a região e o identificador da conta pelos valores corretos.</p>

```bash
aws s3 mb s3://sagemaker-us-east-1-123456789012
```

### 5.7. Enviar o conjunto de dados bruto para o Amazon S3

<p align="justify">Envie o arquivo <code>titanic.csv</code> para o bucket escolhido, mantendo-o no caminho <code>titanic/titanic.csv</code>.</p>

```bash
aws s3 cp data/titanic.csv s3://$(aws s3api list-buckets --query "Buckets[?starts_with(Name, 'sagemaker-')].Name | [0]" --output text)/titanic/titanic.csv
```

### 5.8. Executar o pipeline do orquestrador

<p align="justify">Execute o script orquestrador para construir, atualizar e iniciar o pipeline do SageMaker na AWS.</p>

```bash
python orchestrator.py
```

---

## 6. Monitoramento e Validação

<p align="justify">Console da AWS: acesse <strong>Amazon SageMaker &gt; Pipelines</strong> para acompanhar o progresso da execução do pipeline, os registros das etapas e as métricas disponíveis.</p>

<p align="justify">Registro de Modelos: acesse <strong>Amazon SageMaker &gt; Models &gt; Model Registry</strong> e consulte o grupo <code>titanic-model-group</code> para revisar a nova versão do modelo registrada com o status <code>PendingManualApproval</code>.</p>

---

## 7. Limpeza dos Recursos e Prevenção de Custos

<p align="justify">Ao terminar os testes, remova os recursos que não serão mais utilizados. A execução do pipeline, os artefatos armazenados no Amazon S3, os modelos registrados e outros recursos criados podem permanecer na conta e gerar custos de armazenamento ou computação. Os comandos abaixo são exemplos: substitua os nomes pelos recursos realmente criados e confira cada recurso antes de excluí-lo.</p>

### 7.1. Definir a região e consultar os recursos

```bash
export AWS_REGION="us-east-1"
aws sts get-caller-identity
aws sagemaker list-pipelines --region "$AWS_REGION"
aws sagemaker list-model-package-groups --region "$AWS_REGION"
aws s3 ls
```

### 7.2. Excluir o pipeline do SageMaker

<p align="justify">Se o pipeline foi criado com o nome <code>titanic-pipeline</code>, exclua sua definição após confirmar que não será mais necessária. A exclusão da definição não apaga automaticamente todos os artefatos que as execuções tenham gravado no S3.</p>

```bash
aws sagemaker delete-pipeline \
  --pipeline-name titanic-pipeline \
  --region "$AWS_REGION"
```

### 7.3. Excluir o grupo de pacotes do Model Registry

<p align="justify">Se o projeto criou o grupo <code>titanic-model-group</code>, liste os pacotes registrados e exclua primeiro cada pacote de modelo, caso existam. Use o comando de exclusão do grupo somente depois de remover seus pacotes. Confira os identificadores antes de executar os comandos.</p>

```bash
aws sagemaker list-model-packages \
  --model-package-group-name titanic-model-group \
  --region "$AWS_REGION"
```

<p align="justify">Para cada pacote que não precise mais ser mantido, substitua <code>VERSAO_DO_PACOTE</code> pelo número de versão retornado:</p>

```bash
aws sagemaker delete-model-package \
  --model-package-arn "ARN_DO_PACOTE_DE_MODELO" \
  --region "$AWS_REGION"
```

<p align="justify">Depois de remover todos os pacotes do grupo, exclua o grupo de pacotes:</p>

```bash
aws sagemaker delete-model-package-group \
  --model-package-group-name titanic-model-group \
  --region "$AWS_REGION"
```

### 7.4. Remover artefatos do Amazon S3

<p align="justify">Identifique o bucket e o prefixo utilizados por este projeto. Primeiro, liste os objetos para confirmar que pertencem ao pipeline Titanic. O comando de remoção abaixo apaga permanentemente os objetos dentro do prefixo indicado; não o execute se houver dados de outros projetos nesse local.</p>

```bash
aws s3 ls "s3://NOME_DO_BUCKET/titanic/" --recursive
aws s3 rm "s3://NOME_DO_BUCKET/titanic/" --recursive
```

<p align="justify">Os dados de entrada, os dados processados e os artefatos do treinamento podem estar em prefixos diferentes. Consulte também o bucket e os caminhos indicados nas saídas das execuções do SageMaker antes de apagar outros objetos. Não remova um bucket compartilhado inteiro apenas para limpar este projeto.</p>

### 7.5. Verificar endpoints e modelos implantados

<p align="justify">Se você criou endpoints de inferência durante os testes, eles podem continuar cobrando enquanto estiverem ativos. Liste os endpoints e exclua somente aqueles criados para este projeto. Excluir um endpoint não exclui automaticamente a configuração do endpoint nem o modelo correspondente.</p>

```bash
aws sagemaker list-endpoints --region "$AWS_REGION"
```

<p align="justify">Para cada endpoint do Titanic, substitua os nomes abaixo pelos valores reais. Exclua o endpoint antes de sua configuração e do modelo associado.</p>

```bash
aws sagemaker delete-endpoint \
  --endpoint-name NOME_DO_ENDPOINT \
  --region "$AWS_REGION"

aws sagemaker delete-endpoint-config \
  --endpoint-config-name NOME_DA_CONFIGURACAO \
  --region "$AWS_REGION"

aws sagemaker delete-model \
  --model-name NOME_DO_MODELO \
  --region "$AWS_REGION"
```

### 7.6. Remover a função do IAM criada exclusivamente para o projeto

<p align="justify">Só execute esta etapa se a função <code>AmazonSageMaker-ExecutionRole-Titanic</code> tiver sido criada exclusivamente para este projeto e não for utilizada por outros recursos. Remova a política anexada antes de excluir a função.</p>

```bash
aws iam detach-role-policy \
  --role-name AmazonSageMaker-ExecutionRole-Titanic \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam delete-role \
  --role-name AmazonSageMaker-ExecutionRole-Titanic
```

<p align="justify">Por fim, consulte novamente os recursos no SageMaker e no S3. A limpeza deve ser feita com atenção: não exclua buckets, funções do IAM, modelos ou endpoints compartilhados por outros projetos. A AWS pode manter logs no CloudWatch e objetos em outros prefixos; verifique-os separadamente se desejar remover todos os vestígios do experimento.</p>

---

## 8. Análise de Conformidade com o SOLID

<p align="justify">A estrutura orientada a objetos do projeto aplica os cinco princípios SOLID da seguinte forma:</p>

### 8.1. S — Princípio da Responsabilidade Única

<p align="justify">Cada arquivo e classe possui um único objetivo bem definido. O arquivo <code>preprocessing.py</code>, por meio da classe <code>TitanicDataTransformer</code>, trata apenas da transformação dos dados de entrada. O arquivo <code>train.py</code>, por meio da classe <code>ReproducibleTrainer</code>, cuida do ajuste do modelo e da reprodutibilidade. O arquivo <code>inference.py</code>, por meio da classe <code>InferenceService</code>, cuida da predição e das explicações SHAP. Por fim, o arquivo <code>orchestrator.py</code>, por meio da classe <code>SageMakerPipelineBuilder</code>, orquestra os recursos de infraestrutura na AWS.</p>

### 8.2. O — Princípio Aberto/Fechado

<p align="justify">O código é aberto para expansão, mas fechado para modificação. Por exemplo, em <code>preprocessing.py</code>, para usar o código com outro conjunto de dados, como o Iris, não é necessário alterar a lógica de <code>ProcessingPipeline</code>. Basta criar uma nova subclasse de <code>DataTransformerInterface</code>, como <code>IrisDataTransformer</code>, e passá-la para o pipeline.</p>

### 8.3. L — Princípio da Substituição de Liskov

<p align="justify">Qualquer classe descendente de <code>DataTransformerInterface</code> garante a existência e o comportamento do método <code>fit_transform()</code>. A classe abstrata permite substituir qualquer transformador concreto sem quebrar a execução do pipeline.</p>

### 8.4. I — Princípio da Segregação de Interfaces

<p align="justify">Os trabalhos isolados de processamento, treinamento e inferência não compartilham código desnecessário nem dependências que não utilizam. A função de inferência, em <code>inference.py</code>, não importa bibliotecas pesadas de treinamento, e o script de treinamento não importa o SHAP.</p>

### 8.5. D — Princípio da Inversão de Dependência

<p align="justify">As rotinas dependem de abstrações em vez de implementações concretas. O <code>ProcessingPipeline</code> depende da interface <code>DataTransformerInterface</code>, e não da classe concreta <code>TitanicDataTransformer</code>. Além disso, as etapas do SageMaker dependem de referências abstratas do S3, como <code>transform_step.properties...</code>, permitindo alterar caminhos e clusters sem modificar o código interno.</p>
