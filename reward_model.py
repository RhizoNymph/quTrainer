import polars as pl
import numpy as np
import json
import logging
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

from typing import Any, Tuple, Dict
from math import floor
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, queries: list[str], scores: list[float], tokenizer, max_len: int) -> None:
        self.queries = queries
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self) -> int:
        return len(self.queries)
    def __getitem__(self, i):
        enc = self.tokenizer(
            self.queries[i],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.scores[i], dtype=torch.float)
        return item

class quRewardModel(nn.Module):
    def __init__(self,
        model_id: str = "answerdotai/ModernBERT-base",
        dropout: float = 0.1,
        hidden_dims: list[int] | None = None,
        activation: str = 'relu',
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()

        self.pretrained: Any = AutoModel.from_pretrained(model_id)
        self.hidden_size: int = self.pretrained.config.hidden_size

        if hidden_dims is None:
            hidden_dims = [self.hidden_size // 2]
        
        activation_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }.get(activation, nn.ReLU())

        layers = []
        input_dim = self.hidden_size

        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))
            input_dim = dim

        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(*layers)        
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained(
            input_ids,
            attention_mask,
            return_dict=True
        )

        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        hidden = self.regressor(pooled_output)
        score = self.output_layer(hidden)

        return score.squeeze(-1)

class quTrainer:
    def __init__(
        self,
        model: quRewardModel,
        device: torch.device,
        learning_rate: float = 2e-5,
        head_lr_multiplier: float = 10.0,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        self.model = model.to(device)               
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        bert_params = list(model.pretrained.parameters())
        head_params = list(model.regressor.parameters()) + list(model.output_layer.parameters())

        self.optimizer = optim.AdamW([
            {'params': bert_params, 'lr': learning_rate},
            {'params': head_params, 'lr': learning_rate * head_lr_multiplier}
        ], weight_decay = weight_decay)

        self.criterion = nn.MSELoss()

        self.warmup_ratio = warmup_ratio
        self.scheduler: optim.lr_scheduler.LambdaLR | None = None

    def train_epoch(self, dataloader: DataLoader, epoch: int, verbose: bool = True) -> float:
        self.model.train()
        device = next(self.model.parameters()).device
    
        total_loss = 0

        iterator = tqdm(dataloader, desc=f'Training Epoch {epoch}') if verbose else dataloader        

        for batch_idx, batch in enumerate(iterator):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            predictions = self.model(input_ids, attention_mask)
            loss = self.criterion(predictions, targets) / self.gradient_accumulation_steps            

            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()
                
            total_loss += loss.item() * self.gradient_accumulation_steps

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Tuple[float, float]:
        self.model.eval()
        device = next(self.model.parameters()).device
        total_loss = 0
        all_predictions = []
        all_targets = []

        iterator = tqdm(dataloader, desc='Evaluating') if verbose else dataloader
        
        with torch.no_grad():
            for batch in iterator:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['labels'].to(device)

                predictions = self.model(input_ids, attention_mask)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        predictions_array = np.asarray(all_predictions)
        targets_array = np.asarray(all_targets)
        mae = float(np.mean(np.abs(predictions_array - targets_array)))

        return avg_loss, mae

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 3,
        save_path: str | None = None,
        verbose: bool = True,
        early_stopping_patience: int = 3
    ) -> float:
        best_val_loss = float('inf')
        patience_counter = 0

        total_steps = len(train_dataloader) * num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = self.get_linear_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        for epoch in range(1, num_epochs + 1):
            if verbose: print(f'\n--- Epoch {epoch/num_epochs} ---')

            train_loss = self.train_epoch(train_dataloader, epoch, verbose)
            if verbose:
                print(f'Training Loss: {train_loss:.4f}')

            val_loss, val_mae = self.evaluate(val_dataloader, verbose)
            if verbose:
                print(f'Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_mae': val_mae
                    }, save_path)
                    if verbose:
                        print(f'Saved best model with validation loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f'Early stopping triggered after {epoch} epochs')
                    break
        
        return best_val_loss
    
    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class BayesianOptimizer:
    def __init__(
        self,
        train_queries: list[str],
        train_scores: list[float],
        val_queries: list[str],
        val_scores: list[float],
        tokenizer,
        device: torch.device,
        n_trials: int = 20,
        timeout: int | None = None
    ):
        self.train_queries = train_queries
        self.train_scores = train_scores
        self.val_queries = val_queries
        self.val_scores = val_scores
        self.tokenizer = tokenizer
        self.device = {'device': device}
        self.n_trials = n_trials
        self.timeout = timeout

    def objective(self, trial: Trial) -> float:
        hp_dict = {
            'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 3),
            'hidden_dim_ratio': trial.suggest_float('hidden_dim_ratio', 0.25, 0.75, step=0.25),
            'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'tanh', 'leaky_relu']),
            'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),

            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
            'head_lr_multiplier': trial.suggest_float('head_lr_multiplier', 1.0, 51.0, step=5.0),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.1, step=0.01),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.3, step=0.05),
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4]),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 2.0, step=0.5),
            'max_length': trial.suggest_categorical('max_length', [64, 128, 256]),
            'num_epochs': trial.suggest_int('num_epochs', 2, 10)
        }

        hidden_dims = []
        base_dim = 768
        for i in range(int(hp_dict['num_hidden_layers'])):
            dim = int(base_dim * float(hp_dict['hidden_dim_ratio']) * (0.5 ** i))
            hidden_dims.append(dim)

        train_dataset = QueryDataset(
            self.train_queries, self.train_scores, self.tokenizer, int(hp_dict['max_length'])
        )

        val_dataset = QueryDataset(
            self.val_queries, self.val_scores, self.tokenizer, int(hp_dict['max_length'])
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=int(hp_dict['batch_size']),
            shuffle=True,
            num_workers=0
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(hp_dict['batch_size']),
            shuffle=False,
            num_workers=0
        )

        model = quRewardModel(
            dropout=float(hp_dict['dropout']),
            hidden_dims=hidden_dims,
            activation=str(hp_dict['activation']),
            use_layer_norm=bool(hp_dict['use_layer_norm'])
        )

        trainer = quTrainer(
            model=model,
            device=self.device['device'],
            learning_rate=float(hp_dict['learning_rate']),
            head_lr_multiplier=float(hp_dict['head_lr_multiplier']),
            weight_decay=float(hp_dict['weight_decay']),
            warmup_ratio=float(hp_dict['warmup_ratio']),
            gradient_accumulation_steps=int(hp_dict['gradient_accumulation_steps']),
            max_grad_norm=float(hp_dict['max_grad_norm'])
        )

        val_loss = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=int(hp_dict['num_epochs']),
            save_path=None,
            verbose=False,
            early_stopping_patience=2
        )

        del model
        del trainer
        torch.cuda.empty_cache()

        return val_loss

    def optimize(self) -> Dict[str, Any]:
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            study_name='modernbert-base_quRewardModel'
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"Best validation loss: {best_value:.4f}")
        logger.info(f"Best params: {best_params}")

        history = {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'optimization_history': [
                {
                    'trial': i,
                    'value': trial.value,
                    'params': trial.params
                }
                for i, trial in enumerate(study.trials)
                if trial.value is not None
            ]
        }

        return history

def train_with_best_hyperparameters(
    train_queries: list[str],
    train_scores: list[float],
    val_queries: list[str],
    val_scores: list[float],
    best_params: Dict[str, Any],
    device: torch.device,
    save_path: str = 'best_modernbert_quRewardModel.pt'
):
    logger.info("Training final model with best hyperparameters...")
    
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    
    hidden_dims = []
    base_dim = 768
    for i in range(best_params['num_hidden_layers']):
        dim = int(base_dim * best_params['hidden_dim_ratio'] * (0.5 ** i))
        hidden_dims.append(dim)
    
    train_dataset = QueryDataset(
        train_queries, train_scores, tokenizer, best_params['max_length']
    )
    val_dataset = QueryDataset(
        val_queries, val_scores, tokenizer, best_params['max_length']
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=best_params['batch_size'] * 2,
        shuffle=False,
        num_workers=2
    )
    
    model = quRewardModel(
        dropout=best_params['dropout'],
        hidden_dims=hidden_dims,
        activation=best_params['activation'],
        use_layer_norm=best_params['use_layer_norm']
    )
    
    trainer = quTrainer(
        model=model,
        device=device,
        learning_rate=best_params['learning_rate'],
        head_lr_multiplier=best_params['head_lr_multiplier'],
        weight_decay=best_params['weight_decay'],
        warmup_ratio=best_params['warmup_ratio'],
        gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
        max_grad_norm=best_params['max_grad_norm']
    )
    
    val_loss = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=best_params['num_epochs'],
        save_path=save_path,
        verbose=True,
        early_stopping_patience=3
    )

    if save_path:
        torch.save(model.state_dict(), save_path)
    
    return model, val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # dev_df = pl.read_csv("./data/query-wellformedness/dev.tsv", separator='\t', has_header=False, new_columns=["query", "score"])
    train_df = pl.read_csv("./data/query-wellformedness/train.tsv", separator='\t', has_header=False, new_columns=["query", "score"])
    test_df = pl.read_csv("./data/query-wellformedness/test.tsv", separator='\t', has_header=False, new_columns=["query", "score"])

    queries_list = train_df.select("query").to_series().to_list()
    scores_list = train_df.select("score").to_series().to_list()

    train_len = floor(len(queries_list) * 0.8)

    train_queries = queries_list[:train_len]
    val_queries = queries_list[train_len:]

    train_scores = scores_list[:train_len]
    val_scores = scores_list[train_len:]

    test_queries = test_df.select("query").to_series().to_list()
    test_scores = test_df.select("score").to_series().to_list()

    print("Starting Bayesian Hyperparameter Optimization")

    if not os.path.exists('optimization_history.json'):
        optimizer = BayesianOptimizer(
            train_queries=train_queries,
            train_scores=train_scores,
            val_queries=val_queries,
            val_scores=val_scores,
            tokenizer=tokenizer,
            device=device,
            n_trials=20
        )

        optimization_history = optimizer.optimize()

        with open('optimization_history.json', 'w') as f:
            json.dump(optimization_history, f, indent=2)

    final_model, final_val_loss = train_with_best_hyperparameters(
        train_queries=train_queries,
        train_scores=train_scores,
        val_queries=test_queries,
        val_scores=test_scores,
        best_params=optimization_history['best_params'],
        device=device,
    )
    
    print(f'\nFinal validation loss: {final_val_loss:.4f}')
    print('\nOptimization complete! Best hyperparameters saved to optimization_history.json')
    
    final_model.eval()
    test_query = "How does neural network training work?"
    with torch.no_grad():
        encoding = tokenizer(
            test_query,
            truncation=True,
            padding='max_length',
            max_length=optimization_history['best_params']['max_length'],
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        score = final_model(input_ids, attention_mask)
        print(f'\nTest query: "{test_query}"')
        print(f'Predicted score: {score.item():.4f}')

def analyze_optimization_results(history_path: str = 'optimization_history.json'):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    trials = history['optimization_history']
    trial_numbers = [t['trial'] for t in trials]
    trial_values = [t['value'] for t in trials]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(trial_numbers, trial_values, 'bo-', alpha=0.6, label='Trial values')
    plt.axhline(y=history['best_value'], color='r', linestyle='--', label=f'Best value: {history["best_value"]:.4f}')
    plt.xlabel('Trial')
    plt.ylabel('Validation Loss')
    plt.title('Optimization History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(trial_numbers, np.minimum.accumulate(trial_values).tolist(), 'g-', linewidth=2)
    plt.xlabel('Trial')
    plt.ylabel('Best Validation Loss So Far')
    plt.title('Convergence Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nParameter Analysis:")
    print("-" * 40)
    
    param_values = {}
    for trial in trials:
        for param, value in trial['params'].items():
            if param not in param_values:
                param_values[param] = []
            param_values[param].append(value)
    
    for param, values in param_values.items():
        if all(isinstance(v, (int, float)) for v in values):
            correlation = np.corrcoef(values, trial_values)[0, 1]
            print(f"{param}: correlation with loss = {correlation:.3f}")
    
    print("\nBest Hyperparameters:")
    print("-" * 40)
    for param, value in history['best_params'].items():
        print(f"{param}: {value}")

if __name__ == '__main__':
    main()
    analyze_optimization_results()