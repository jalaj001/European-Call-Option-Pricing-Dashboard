# Quant Option Pricing Dashboard

A Python-based quantitative finance application for calculating option prices using multiple pricing models.

## Features

- **Black-Scholes Model**: Analytical option pricing for European vanilla options
- **Binomial Tree Model (CRR)**: Discrete-time option pricing using Cox-Ross-Rubinstein approach
- **Monte Carlo Simulation**: Stochastic option pricing using Monte Carlo methods

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Quant-Option-Pricing-Dashboard.git
cd Quant-Option-Pricing-Dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run test1.py
```

## Pricing Models

### Black-Scholes
Closed-form solution for European option pricing. Assumes no arbitrage and constant volatility.

### Binomial Model
Recursive approach that builds a tree of possible price movements. More flexible than Black-Scholes for American options.

### Monte Carlo
Simulates thousands of possible price paths to estimate option values. Useful for complex derivatives.

## Files

- `test.py` - Core option pricing implementations
- `test1.py` - Streamlit dashboard application
- `requirements.txt` - Python dependencies

## License

MIT License
