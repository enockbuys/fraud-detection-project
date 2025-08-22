Fraud Detection Using Generative AI
Project Overview
This project implements a fraud detection system that uses Conditional Generative Adversarial Networks (CGANs) to generate synthetic fraudulent transaction data, addressing class imbalance issues in financial datasets. The system combines synthetic data generation with Random Forest classification to improve fraud detection performance.

Project Structure
text
fraud-detection-project/
├── .idea/                          # IDE configuration files
├── __pycache__/                    # Python cache files
├── data/                           # Dataset directory
├── classification_summary.py       # Classification metrics calculator
├── conditional_gan.py              # Conditional GAN implementation
├── db_manager.py                   # Database management utilities
├── fraud_detection.py              # Main fraud detection module
├── main.py                         # Main execution script
├── my_decision_tree.py             # Custom decision tree implementation
├── my_random_forest.py             # Custom random forest implementation
├── preprocessor.py                 # Data preprocessing utilities
├── schema.sql                      # Database schema
└── UI.py                           # Graphical user interface
Installation
1. Ensure you have Python 3.7+ installed
2. Install required dependencies:
  pip install numpy pandas scikit-learn tkinter sqlite3

Usage
1. Place your transaction data in CSV format in the data/ directory
2. Run the main application:
  python UI.py
3. Alternatively, run the command-line version:
  python main.py

Key Features
1. Data preprocessing and balancing
2. Conditional GAN implementation for synthetic data generation
3. Custom Random Forest classifier
4. SQLite database integration for results storage
5. Graphical user interface for easy interaction
6. Support for multiple synthetic data augmentation percentages (5%, 10%, 15%)

File Descriptions
1. classification_summary.py: Calculates and displays classification metrics
2. conditional_gan.py: Implements Conditional GAN for synthetic data generation
3. db_manager.py: Manages SQLite database operations
4. my_decision_tree.py: Custom decision tree implementation
5. my_random_forest.py: Custom random forest implementation
6. preprocessor.py: Handles data cleaning and preparation
7. UI.py: Graphical user interface for the application
8. main.py: Command-line interface for the system
9. schema.sql: Database schema definition

Results
The system evaluates fraud detection performance using:
  Accuracy, Precision, Recall, and F1-score
  Confusion matrix analysis
  Comparison between different synthetic data augmentation levels

Future Work
  Implement additional generative models (VAEs)
  Add support for real-time fraud detection
  Enhance model explainability features
  Expand database support beyond SQLite

Notes
The project uses custom implementations of decision trees and random forests
The Conditional GAN is implemented from scratch without relying on deep learning frameworks
The system is designed to work with financial transaction data in CSV format
