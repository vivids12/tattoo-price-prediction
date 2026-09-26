# tattoo-price-prediction

Estrutura inicial de um projeto de Machine Learning em Python para, futuramente, prever:

- preço sugerido de uma tatuagem;
- tempo estimado para realização da tatuagem.

> Neste momento, o projeto contém apenas a organização base. Não há implementação de treino, integração com MySQL, Spring Boot, React Native ou APIs externas.

## Estrutura do projeto

```text
tattoo-price-prediction/
├── data/
├── models/
├── src/
│   ├── preprocessing/
│   ├── prediction/
│   └── training/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Finalidade de cada pasta e arquivo

- `data/`: diretório reservado para dados de entrada e artefatos intermediários de dados.
- `models/`: diretório reservado para armazenamento de modelos treinados e artefatos relacionados.
- `src/`: código-fonte principal do projeto.
  - `src/preprocessing/`: módulos de preparação e transformação de dados (futuro).
  - `src/prediction/`: módulos de inferência/predição (futuro).
  - `src/training/`: módulos do pipeline de treinamento e avaliação (futuro).
- `tests/`: diretório reservado para testes automatizados.
- `requirements.txt`: dependências Python iniciais para evolução do projeto de ML.
- `README.md`: documentação principal e visão geral da solução.
- `.gitignore`: regras para evitar versionar arquivos temporários, ambientes virtuais e artefatos de build.

## Arquivos iniciais de código (placeholders)

Foram criados módulos iniciais com funções placeholder (`NotImplementedError`) para delimitar responsabilidades futuras:

- `src/preprocessing/pipeline.py`
- `src/prediction/service.py`
- `src/training/pipeline.py`

Esses pontos de entrada ajudam a organizar a evolução para o cenário futuro com modelo Random Forest e integração ao back-end Java/Spring Boot.
