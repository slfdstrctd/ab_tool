# A/B Testing Tool

Automated statistical analysis system for A/B experiments that processes user behavior data and makes data-driven decisions.

## Approach

The system uses multiple statistical methods to analyze experiments:

1. **Bootstrap resampling** - 10000 iterations
2. **T-test** (parametric)
3. **Mann-Whitney U** (non-parametric)

Decision criteria:
- **ACCEPT**: p-value < 0.12, positive effect, adequate sample size
- **REJECT**: No significant positive effects or negative effects detected
- **KEEP RUNNING**: Inconclusive results or insufficient sample size

## Installation

```bash
pip install -r requirements.txt
```

### Data
Put CSV files into `./data` folder.

The system expects:
- **Users files**: user_id, ts, and ampl_user_data (JSON)
- **Messages files**: user_id and messages_count
- **Payments files**: user_id and price_usd


## Usage

### Web Interface
```bash
streamlit run web_app.py
```

### Command Line
```bash
# Analyze experiment (e.g. add_bttn_fix)
python main.py -e add_bttn_fix

# Analyze all experiments
python main.py --all-experiments

# Custom parameters
python main.py -e experiment_name --significance-level 0.1 --metrics revenue_usd messages_count
```

### Command Line Options
- `-e, --experiment`: Experiment name
- `-d, --data-dir`: CSV files directory (default: `all_csv_files`)
- `-o, --output-dir`: Output directory (default: `outputs`)
- `-m, --metrics`: Metrics to analyze (default: revenue_usd, messages_count)
- `-s, --significance-level`: P-value threshold (default: 0.12)
- `--effect-sizes`: Effect sizes to test (default: 1.0, 2.0, 5.0, 10.0)
- `--all-experiments`: Analyze all experiments
