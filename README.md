# VishwamAI Datasets

This repository contains datasets for training and evaluating machine learning models for a chat agent. The datasets cover various domains, including room math, MMLU, reasoning, and additional benchmarks. The goal is to achieve 100% accuracy, high quality, quantity, and lack of bias in the datasets.

## Model Training and Evaluation

To train models using these datasets and evaluate their performance against the benchmarks, follow these steps:

### Prepare the Training Environment

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Train the Model

2. Train the model using the `train_model.py` script:
   ```bash
   python train_model.py --dataset combined_dataset.csv
   ```

### Evaluate the Model

3. Evaluate the model using the `evaluate_model.py` script:
   ```bash
   python evaluate_model.py --dataset combined_dataset.csv --model_path path_to_trained_model
   ```

## Accessing Large Files

Due to the size limitations of Git, the large dataset files are not included directly in the repository. Instead, they are available as assets in the GitHub release. You can download the large files from the following links:

- [CommonsenseQA Dataset](https://github.com/VishwamAI/datasets/releases/download/v1.1.1/commonsenseqa.csv)
- [Break Dataset](https://github.com/VishwamAI/datasets/releases/download/v1.1.1/break.csv)
- [bAbI Tasks Dataset](https://github.com/VishwamAI/datasets/releases/download/v1.1.1/babi.csv)
- [Natural Questions Dataset](https://github.com/VishwamAI/datasets/releases/download/v1.1.1/natural_questions.csv)

## Contributing

Contributions are welcome! Please follow the GitHub Pull Request Workflow to contribute to this repository. To contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them with clear and descriptive messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
