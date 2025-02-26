import os
import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import numpy as np

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from different clients using weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "total_examples": sum(examples)
    }

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Create weights directory if it doesn't exist
            os.makedirs('weights', exist_ok=True)
            
            # Save weights
            weights = [np.array(param) for param in aggregated_parameters]
            np.save(f"weights/round_{server_round}_weights.npy", weights)
        
        return aggregated_parameters

def main():
    """Start the Flower server."""
    
    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average
    )

    # Start Flower server
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
