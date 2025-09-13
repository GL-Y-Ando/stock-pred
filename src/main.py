#!/usr/bin/env python3
"""
Stock Price Trend Prediction System
Main entry point for the ML-based stock prediction system

Features:
- Neural network training and prediction
- Short & long trend analysis (5-day & 25-day moving averages)
- Trend reversal price prediction
- Confidence rating system
- Model update capability
- Data splitting and validation
"""

import os
import sys
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

# Core modules
from data_manager import DataManager
from model_trainer import ModelTrainer
from predictor import Predictor
from trend_analyzer import TrendAnalyzer
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPredictionSystem:
    """
    Main class for the stock prediction system
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize the stock prediction system
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = Config(config_path)
        self.data_manager = DataManager(self.config)
        self.trend_analyzer = TrendAnalyzer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.predictor = Predictor(self.config)
        
        logger.info("Stock Prediction System initialized")
    
    def load_data(self):
        """
        Load and prepare stock data
        """
        logger.info("Loading stock data...")
        
        # Load stock list
        stock_list = self.data_manager.load_stock_list()
        logger.info(f"Loaded {len(stock_list)} stocks from stock list")
        
        # Load price data for all stocks
        Limited_stocks = stock_list["Code"].head(50).tolist()
        price_data = self.data_manager.load_price_data(Limited_stocks)
        logger.info(f"Loaded price data for {len(price_data)} stocks")
        
        return stock_list, price_data
    
    def prepare_training_data(self, price_data):
        """
        Prepare data for model training
        
        Args:
            price_data (dict): Stock price data
            
        Returns:
            tuple: Training and testing datasets
        """
        logger.info("Preparing training data...")
        
        # Calculate technical indicators
        processed_data = self.trend_analyzer.calculate_moving_averages(price_data)
        processed_data = self.trend_analyzer.calculate_trend_features(processed_data)
        
        # Split data for training and testing
        train_data, validation_data, test_data = self.data_manager.split_data(
            processed_data, 
            train_ratio=self.config.data.train_ratio
        )
        
        logger.info(f"Training data: {len(train_data)} stocks")
        logger.info(f"Validation data: {len(validation_data)} stocks")
        logger.info(f"Testing data: {len(test_data)} stocks")
        
        return train_data, test_data
    
    def train_models(self, train_data):
        """
        Train the neural network models
        
        Args:
            train_data (dict): Training dataset
        """
        logger.info("Training neural network models...")
        
        # Train short-term trend model
        self.model_trainer.train_short_term_model(train_data)
        logger.info("Short-term trend model trained")
        
        # Train long-term trend model
        self.model_trainer.train_long_term_model(train_data)
        logger.info("Long-term trend model trained")
        
        # Train reversal price prediction models
        self.model_trainer.train_reversal_models(train_data)
        logger.info("Reversal price prediction models trained")
        
        # Save trained models
        self.model_trainer.save_models()
        logger.info("All models saved successfully")
    
    def evaluate_models(self, test_data):
        """
        Evaluate model performance on test data
        
        Args:
            test_data (dict): Testing dataset
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        metrics = self.model_trainer.evaluate_models(test_data)
        
        logger.info("Model evaluation completed:")
        for model_name, model_metrics in metrics.items():
            logger.info(f"{model_name}: {model_metrics}")
        
        return metrics
    
    def make_predictions(self, stock_symbols=None):
        """
        Generate predictions for specified stocks
        
        Args:
            stock_symbols (list): List of stock symbols to predict (None for all)
            
        Returns:
            dict: Predictions for each stock
        """
        logger.info("Generating predictions...")
        
        # If no specific stocks provided, load all stocks from stock list
        if stock_symbols is None:
            try:
                # Load stock list if not already loaded
                if self.data_manager.stock_list is None:
                    stock_list = self.data_manager.load_stock_list()
                    logger.info(f"Loaded stock list with {len(stock_list)} stocks")
                
                # Get stock codes from the stock list
                if hasattr(self.data_manager.stock_list, 'Code'):
                    stock_symbols = self.data_manager.stock_list['Code'].tolist()
                elif hasattr(self.data_manager.stock_list, 'code'):
                    stock_symbols = self.data_manager.stock_list['code'].tolist()
                else:
                    # Fallback: try to discover available stock data files
                    stock_symbols = self.data_manager._discover_available_stocks()
                    
                logger.info(f"Will generate predictions for {len(stock_symbols)} stocks")
                
            except Exception as e:
                logger.error(f"Error loading stock list: {e}")
                return {}
        
        # Load latest data for the specified stocks
        current_data = self.data_manager.get_latest_data(stock_symbols)
        logger.info(f"Loaded data for {len(current_data)} stocks")
        
        if not current_data:
            logger.warning("No stock data loaded - predictions cannot be generated")
            return {}
        
        predictions = {}
        
        for symbol, data in current_data.items():
            try:
                # Calculate current trends
                short_trend = self.trend_analyzer.calculate_short_trend(data)
                long_trend = self.trend_analyzer.calculate_long_trend(data)
                
                # Predict trend reversals
                short_reversal = self.predictor.predict_short_reversal(data)
                long_reversal = self.predictor.predict_long_reversal(data)
                
                # Calculate confidence ratings
                short_confidence = self.predictor.calculate_confidence(data, 'short')
                long_confidence = self.predictor.calculate_confidence(data, 'long')
                
                predictions[symbol] = {
                    'short_trend': short_trend,
                    'long_trend': long_trend,
                    'short_reversal_price': short_reversal['price'],
                    'long_reversal_price': long_reversal['price'],
                    'short_confidence': short_confidence,
                    'long_confidence': long_confidence,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")
                predictions[symbol] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        logger.info(f"Generated predictions for {len(predictions)} stocks")
        return predictions
    
    def update_models(self):
        """
        Update the prediction models with new data
        """
        logger.info("Updating models with latest data...")
        
        # Load latest data
        stock_list, price_data = self.load_data()
        
        # Prepare training data
        train_data, test_data = self.prepare_training_data(price_data)
        
        # Retrain models
        self.train_models(train_data)
        
        # Evaluate updated models
        metrics = self.evaluate_models(test_data)
        
        logger.info("Model update completed")
        return metrics
    
    def run_full_pipeline(self):
        """
        Run the complete training and prediction pipeline
        """
        logger.info("Starting full pipeline...")
        
        try:
            # 1. Load data
            stock_list, price_data = self.load_data()
            
            # 2. Prepare training data
            train_data, test_data = self.prepare_training_data(price_data)
            
            # 3. Train models
            self.train_models(train_data)
            
            # 4. Evaluate models
            metrics = self.evaluate_models(test_data)
            
            # 5. Generate sample predictions
            sample_stocks = list(price_data.keys())[:10]  # Predict for first 10 stocks
            predictions = self.make_predictions(sample_stocks)
            
            logger.info("Full pipeline completed successfully")
            return {
                'metrics': metrics,
                'sample_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """
    Main function - command line interface
    """
    parser = argparse.ArgumentParser(description='Stock Prediction System')
    parser.add_argument('--mode', choices=['train', 'predict', 'update', 'full'], 
                       default='predict', help='Operation mode')
    parser.add_argument('--stocks', nargs='+', help='Stock symbols to predict')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--output', default='predictions.json', help='Output file name (will be saved in output/ directory)')  # Changed default
    
    args = parser.parse_args()
    
    try:
        # Initialize the system
        system = StockPredictionSystem(args.config)
        
        if args.mode == 'train':
            logger.info("Training mode selected")
            stock_list, price_data = system.load_data()
            train_data, test_data = system.prepare_training_data(price_data)
            system.train_models(train_data)
            metrics = system.evaluate_models(test_data)
            print("Training completed. Metrics:", metrics)
            
        elif args.mode == 'predict':
            logger.info("Prediction mode selected")
            predictions = system.make_predictions(args.stocks)
            
            # Ensure output directory exists and use absolute path
            output_file = args.output
            if not os.path.isabs(output_file):
                output_file = os.path.abspath(output_file)
            
            output_dir = os.path.dirname(output_file)
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Created output directory: {output_dir}")
                    print(f"Output directory created: {output_dir}")
                except Exception as e:
                    logger.error(f"Failed to create output directory {output_dir}: {e}")
                    print(f"Warning: Could not create output directory: {e}")
            
            # Save predictions
            import json
            try:
                print(f"Attempting to save to: {output_file}")
                with open(output_file, 'w') as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False)
                
                # Get absolute path for user clarity
                abs_path = os.path.abspath(output_file)
                logger.info(f"Predictions saved to {abs_path}")
                print(f"Predictions saved to {output_file}")
                print(f"Full path: {abs_path}")
                
            except Exception as e:
                logger.error(f"Error saving predictions to {output_file}: {e}")
                print(f"Error saving predictions: {e}")
                sys.exit(1)
            
            # Display sample results
            if predictions:
                print(f"\nGenerated predictions for {len(predictions)} stocks")
                sample_count = 0
                for symbol, pred in predictions.items():
                    if sample_count >= 5:  # Show only first 5
                        break
                    if 'error' not in pred:
                        print(f"\n{symbol}:")
                        print(f"  Short trend: {pred.get('short_trend', 'N/A')}")
                        print(f"  Long trend: {pred.get('long_trend', 'N/A')}")
                        try:
                            short_reversal = pred.get('short_reversal_price', 0)
                            long_reversal = pred.get('long_reversal_price', 0)
                            short_conf = pred.get('short_confidence', 0)
                            long_conf = pred.get('long_confidence', 0)
                            
                            print(f"  Short reversal: {short_reversal:.2f}")
                            print(f"  Long reversal: {long_reversal:.2f}")
                            print(f"  Confidence: {short_conf:.2f}% / {long_conf:.2f}%")
                        except (TypeError, ValueError, AttributeError):
                            print(f"  Short reversal: {pred.get('short_reversal_price', 'N/A')}")
                            print(f"  Long reversal: {pred.get('long_reversal_price', 'N/A')}")
                            print(f"  Confidence: {pred.get('short_confidence', 'N/A')}% / {pred.get('long_confidence', 'N/A')}%")
                        sample_count += 1
                    else:
                        print(f"\n{symbol}: Error - {pred.get('error', 'Unknown error')}")
                        sample_count += 1
                
                if len(predictions) > 5:
                    print(f"\n... and {len(predictions) - 5} more stocks")
                    
                # Show summary statistics
                successful = sum(1 for pred in predictions.values() if 'error' not in pred)
                failed = len(predictions) - successful
                print(f"\nSummary: {successful} successful, {failed} failed predictions")
            else:
                print("No predictions generated")
            
        elif args.mode == 'update':
            logger.info("Update mode selected")
            metrics = system.update_models()
            print("Models updated. New metrics:", metrics)
            
        elif args.mode == 'full':
            logger.info("Full pipeline mode selected")
            results = system.run_full_pipeline()
            print("Full pipeline completed:", results['metrics'])
            
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()