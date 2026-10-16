# Impact of 6G Network Performance on Manufacturing Efficiency in Smart Factories

### An Exploratory Data Analysis and Network–Efficiency Diagnostic Framework for Smart Manufacturing

**Project Context:** Smart Manufacturing Analytics  
**Internship Organization:** Unified Mentor Private Limited  
**Domain:** Manufacturing Analytics | Network Performance Analysis | Exploratory Data Analysis  
**Author:** Isha Kalra | BCA 3rd Year | Dewan V.S. Group of Institutions, Meerut

**Live Dashboard:** [Explore the Interactive Streamlit Application](https://6g-manufacturing-analytics-uwwmlrkw9p7lbgzmjpwmrk.streamlit.app/)

**GitHub Repository:** [View Source Code and Project Artifacts](https://github.com/ikalra858-cell/6g-manufacturing-analytics) 

---

## Overview

In Industry 4.0 and emerging Industry 5.0 environments, connected machines depend on timely data exchange to coordinate operations, monitor production, and support data-driven decisions. Network latency and packet loss are therefore important indicators to examine alongside manufacturing performance and machine-condition metrics.

This project analyzes a dataset of 100,000 manufacturing observations across 50 machines, covering 1 January to 10 March 2025. Using exploratory data analysis, statistical association measures, comparative diagnostics, and an interactive Streamlit dashboard, it examines how network-performance indicators vary alongside production speed, efficiency classifications, defect rates, operational errors, and machine conditions.

The project follows a connectivity-focused approach to manufacturing analytics, with an emphasis on identifying observable patterns, evaluating potential performance concerns, and developing data-informed monitoring recommendations.

---

## Business Problem

Manufacturing organizations need a clearer understanding of how communication-network conditions relate to production performance and quality. Without careful analysis, changes in manufacturing outcomes may be difficult to interpret, and network-related indicators may be overlooked when investigating operational issues.

This project addresses four key questions:

- How do network latency and packet loss vary across the observed manufacturing data?
- Do efficiency classifications, production speed, and quality indicators show consistent patterns across different network-quality levels?
- Are particular operation modes or machine-condition profiles associated with different observed network-efficiency patterns?
- How can network indicators be monitored alongside production, quality, and machine-condition metrics to support better operational investigation?

The analysis is descriptive and diagnostic. It evaluates observed relationships in the available dataset; it does not establish that network conditions caused manufacturing outcomes, prove the performance of a real 6G deployment, or define universal network-engineering thresholds.

---

## Dataset Overview

The project analyzes manufacturing observations recorded across multiple machines and operational conditions. The dataset combines network-performance indicators with production, product-quality, and machine-condition metrics to support an exploratory investigation of network–efficiency relationships.

### Dataset Summary

| Attribute | Details |
|---|---|
| Dataset size | 100,000 observations |
| Machines represented | 50 |
| Number of columns | 14 |
| Observation period | 1 January 2025 – 10 March 2025 |
| Data type | Structured, tabular data |
| Analytical approach | Exploratory and descriptive analysis |

### Key Variables

| Category | Variables | Purpose |
|---|---|---|
| Time and machine identification | `Date`, `Timestamp`, `Machine_ID` | Examine observations over time and across machines |
| Operating conditions | `Operation_Mode` | Compare patterns across operational modes |
| Machine condition | `Temperature_C`, `Vibration_Hz`, `Power_Consumption_kW` | Explore machine-condition and resource-use indicators |
| Network performance | `Network_Latency_ms`, `Packet_Loss_%` | Examine network conditions and their associations with manufacturing metrics |
| Production and quality | `Production_Speed_units_per_hr`, `Quality_Control_Defect_Rate_%`, `Error_Rate_%` | Assess observed production, quality, and error patterns |
| Predictive and efficiency indicators | `Predictive_Maintenance_Score`, `Efficiency_Status` | Explore the supplied maintenance score and efficiency classifications |

### Dataset Scope

The analysis uses the available observations to compare network indicators with manufacturing outcomes across different operating conditions. It focuses on descriptive patterns, statistical associations, group comparisons, and potential monitoring opportunities rather than causal claims.

The dataset's observed relationships should be interpreted within its recorded time period, available variables, and data-quality constraints. The analysis does not independently verify that the observations were collected during a real-world 6G deployment.

---

## Key Findings

The analysis examined the relationships between network-performance indicators and manufacturing outcomes using descriptive statistics, correlation analysis, and comparative diagnostics.

### 1. Network Latency and Production Speed

- The Spearman correlation between network latency and production speed was approximately **−0.0011**, indicating virtually no monotonic association in the analyzed observations.
- The estimated linear slope was approximately **−0.009671 production units per hour per millisecond of latency**. This is a small descriptive relationship in the fitted data and should not be interpreted as evidence of a meaningful operational effect.

**Interpretation:** The available dataset does not show a meaningful monotonic relationship between latency and production speed. Further investigation would be needed to determine whether other variables or operating conditions affect this relationship.

### 2. Packet Loss and Operational Error Rate

- The Spearman correlation between packet loss and error rate was approximately **−0.0024**, indicating virtually no monotonic association.
- The analysis did not identify a consistent increase in operational error rate as packet loss increased.

**Interpretation:** The observed data does not support a clear monotonic relationship between packet loss and operational errors. This does not establish that packet loss is harmless in other manufacturing environments.

### 3. Production Speed Across Packet-Loss Deciles

- The mean production speed in the lowest packet-loss decile (D1) was approximately **278.10 units/hour**.
- The mean production speed in the highest packet-loss decile (D10) was approximately **273.81 units/hour**.
- The calculated Packet Loss Impact Ratio was approximately **1.54%**, representing the relative difference between these two mean production-speed values.
- The intermediate deciles did not follow a consistently increasing or decreasing pattern.

**Interpretation:** The difference between the lowest and highest packet-loss groups was modest, and the non-monotonic pattern limits the conclusions that can be drawn from this comparison. The ratio is a descriptive comparison, not a causal estimate of packet loss impact.

### 4. Quality and Efficiency Patterns

- Defect-rate and error-rate indicators remained relatively stable across the comparisons performed.
- The analysis did not establish a consistent network-sensitive machine group or a universal network threshold associated with lower manufacturing efficiency.

**Interpretation:** The observed relationships are not strong enough to justify a universal operational rule. Network metrics should be assessed alongside machine condition, operation mode, production performance, and data quality.

### Overall Takeaway

Within this dataset, the investigated network indicators showed little measurable monotonic association with the selected manufacturing outcomes. These results highlight the importance of validating assumptions with data, examining multiple operational factors, and avoiding causal conclusions from observational analysis.

---

## Core Metrics and Definitions

The project uses network, production, and quality indicators to investigate observed patterns across the manufacturing dataset.

| Metric | Definition / Interpretation |
|---|---|
| **Network Latency (ms)** | The recorded time delay associated with network communication. |
| **Packet Loss (%)** | The percentage of network packets reported as lost. |
| **Network Stability Index (NSI)** | A project-defined composite indicator calculated from normalized latency and packet loss. Higher values indicate better relative network stability under this project's formula. |
| **Production Speed (units/hour)** | The recorded number of production units per hour. |
| **Quality Control Defect Rate (%)** | The recorded proportion of output classified as defective. |
| **Error Rate (%)** | The recorded operational error-rate indicator. |
| **Predictive Maintenance Score** | The dataset-provided score used to explore potential machine-maintenance patterns. |
| **Efficiency Status** | The efficiency classification supplied in the dataset. |

### Network Stability Index (NSI)

The Network Stability Index is defined for this project as:

\[
\text{NSI} = 1 - \left(0.5L_n + 0.5P_n\right)
\]

Where:

- \(L_n\) is normalized network latency.
- \(P_n\) is normalized packet loss.
- Both indicators are normalized using the minimum and maximum values from the full dataset.

The index assigns equal weight to normalized latency and packet loss. Under this formula, a higher NSI represents better relative network stability within the analyzed data.

**Important:** NSI is a project-defined analytical metric, not an established industry-standard 6G measurement. Its interpretation depends on the normalization range and weighting choices used in this analysis.

---

## Methodology

The project follows a structured exploratory data analysis (EDA) workflow to examine network-performance indicators alongside manufacturing outcomes.

### 1. Data Inspection and Preparation

- Loaded and inspected the manufacturing dataset using Pandas.
- Examined the dataset's dimensions, column names, data types, and available time range.
- Reviewed relevant network, production, quality, machine-condition, and efficiency variables.
- Considered data completeness and timestamp consistency when interpreting results.

### 2. Exploratory Data Analysis

- Examined distributions and summary statistics for key numerical variables.
- Compared network-performance indicators across recorded operating conditions.
- Investigated production speed, defect rates, error rates, and efficiency classifications.
- Used visualizations to identify patterns, variation, and potential anomalies.

### 3. Statistical Association Analysis

- Used Spearman rank correlation to examine monotonic relationships between network indicators and manufacturing outcomes.
- Examined the descriptive linear relationship between network latency and production speed.
- Compared production speed across packet-loss deciles to assess whether the observed pattern was consistent across increasing packet-loss groups.
- Interpreted statistical results in context rather than treating association as evidence of causation.

### 4. Comparative Diagnostics

- Compared outcomes across operation modes and available efficiency classifications.
- Examined network indicators alongside machine-condition and quality metrics.
- Investigated whether the available comparisons supported consistent group-level patterns or clear operational thresholds.

### 5. Network Stability Index Development

- Normalized network latency and packet loss using the full dataset's minimum and maximum values.
- Combined the normalized indicators with equal weighting to calculate the project-defined Network Stability Index (NSI).
- Used the index as a comparative analytical indicator, not as an industry-standard network measurement.

### 6. Interactive Dashboard Development

- Built an interactive dashboard using Streamlit and Plotly.
- Added filters to support focused exploration of dates, network quality, efficiency classifications, and operation modes.
- Presented summary indicators, comparisons, and visual diagnostics to make the analysis easier to explore.

### 7. Interpretation and Validation

- Reviewed findings against the calculated statistical results and dashboard comparisons.
- Considered timestamp-related data-quality limitations when interpreting time-based patterns.
- Distinguished descriptive findings from causal claims and documented the limitations of the available dataset.

---

## Interactive Dashboard Features

The project includes an interactive Streamlit dashboard that presents manufacturing and network-performance analysis through seven dedicated tabs. Interactive filters allow users to narrow the analysis and investigate different subsets of the available observations.

### Dashboard Tabs

| Tab | Purpose |
|---|---|
| **1. Overview** | Presents high-level KPIs and a summary of the selected manufacturing observations. |
| **2. Network × Efficiency** | Explores network-performance indicators alongside manufacturing efficiency classifications. |
| **3. Latency Impact** | Examines the observed relationship between network latency and production speed. |
| **4. Quality & Errors** | Compares quality-control defect rates and operational error indicators. |
| **5. Operation Modes** | Supports comparisons of network and manufacturing patterns across recorded operation modes. |
| **6. 6G Insights** | Presents network-related analytical observations and potential monitoring considerations. |
| **7. Method & Limits** | Explains the analytical approach, interpretation boundaries, and limitations of the project. |

### Interactive Filters

The dashboard provides filters for:

- **Date range:** Focus on observations within a selected period.
- **Network quality:** Explore subsets based on network-quality indicators.
- **Efficiency classification:** Compare observations associated with selected efficiency categories.
- **Operation mode:** Investigate recorded operating conditions separately.

### Key Dashboard Capabilities

- KPI cards for indicators such as Network Stability Index (NSI), latency, packet loss, production speed, and defect rate.
- Interactive charts built with Plotly.
- Filter-driven comparisons to support focused exploratory analysis.
- Sample-size warnings to help users interpret filtered results with appropriate caution.
- Methodology and limitation notes to distinguish descriptive observations from causal conclusions.

**Open the live dashboard:** [6G Manufacturing Analytics — Streamlit App](https://6g-manufacturing-analytics-uwwmlrkw9p7lbgzmjpwmrk.streamlit.app/)

---

## Technology Stack

| Technology | Role in the Project |
|---|---|
| **Python** | Core programming language for data analysis and dashboard logic |
| **Pandas** | Data loading, manipulation, aggregation, and analysis |
| **NumPy** | Numerical operations and metric calculations |
| **Matplotlib & Seaborn** | Exploratory data visualizations |
| **Plotly** | Interactive charts in the dashboard |
| **Streamlit** | Interactive dashboard development and deployment |
| **Jupyter Notebook** | Exploratory analysis, statistical investigation, and documentation |
| **Git & GitHub** | Version control and project hosting |
| **Streamlit Community Cloud** | Hosting the live interactive dashboard |

---

## Repository Structure

```text
6g-manufacturing-analytics/
│
├── dashboard/
│   ├── app.py
│   └── requirements.txt
│
├── data/
│   └── raw/
│       └── Thales_Group_Manufacturing.csv
│
├── notebooks/
│   └── 02_6G_Manufacturing_Analytics_Analysis.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── F01_Network_Performance_Distributions.png
│   │   ├── F02_Daily_Network_Stability_Index.png
│   │   ├── F03_Efficiency_by_Network_Quality.png
│   │   ├── F04_Production_Speed_vs_Latency.png
│   │   ├── F05_Production_Speed_by_Latency_Decile.png
│   │   ├── F06_Error_Rate_vs_Packet_Loss.png
│   │   ├── F07_Defect_Rate_by_Packet_Loss_Range.png
│   │   └── F08_Production_Speed_by_Latency_and_Operation_Mode.png
│   │
│   └── tables/
│       ├── T01_Final_Source_of_Truth_Metrics.csv
│       ├── T02_Network_Quality_vs_Efficiency.csv
│       ├── T03_Latency_Decile_Performance.csv
│       ├── T04_Packet_Loss_Reliability.csv
│       ├── T05_Operation_Mode_Summary.csv
│       └── T06_Comparable_Mechanical_Condition_Efficiency.csv
│
├── research/
│   ├── Research paper.pdf
│   └── Executive Summary.pdf
│
├── src/
│
├── .gitignore
└── README.md
```

### Directory Guide

- **`dashboard/`** — Streamlit application and dashboard dependencies.
- **`data/raw/`** — Original manufacturing dataset.
- **`notebooks/`** — Exploratory data analysis and statistical investigation.
- **`outputs/figures/`** — Eight exported visualizations generated during the analysis.
- **`outputs/tables/`** — Six exported analytical tables.
- **`research/`** — Research paper and executive summary, to be added when finalized.
- **`src/`** — Location for reusable source code.

---

## Data Quality and Validation

Data quality was considered during the analysis to help ensure that the observed patterns were interpreted appropriately.

### Validation Considerations

- **Dataset structure:** Inspected the dataset dimensions, column names, and data types before analysis.
- **Timestamp consistency:** Investigated the temporal structure of the recorded observations.
- **Statistical consistency:** Compared summary statistics, correlation results, and group-level comparisons when interpreting findings.
- **Filtered sample sizes:** Incorporated sample-size warnings into the dashboard to encourage caution when exploring smaller subsets.
- **Interpretation boundaries:** Distinguished descriptive associations from causal conclusions.

### Timestamp Anomaly

The analysis identified an unusual timestamp pattern on **1 March 2025**, involving 2,880 observations with a repeated minute sequence. These observations were retained in the dataset, but the anomaly warrants caution when interpreting time-based patterns.

The observation gaps in the recorded timestamps represent intervals between observations; they should not automatically be interpreted as network delays or outages.

### Validation Limitations

These checks support more careful interpretation of the available data, but they do not independently verify the dataset's collection process, establish that it represents a real-world 6G deployment, or guarantee that every recorded measurement is accurate.

---

## Recommendations and Practical Implications

The findings suggest that network-performance indicators should be evaluated alongside production, quality, and machine-condition metrics rather than interpreted in isolation.

Based on the descriptive analysis, the following recommendations may support further investigation in smart-manufacturing environments.

### 1. Monitor Network and Production Metrics Together

Track network latency and packet loss alongside production speed, defect rate, and operational error indicators. Reviewing these metrics together can help analysts investigate whether changes occur at the same time.

### 2. Investigate Operational Context

Compare network and manufacturing patterns across operation modes and machine-condition profiles. Contextual comparisons may help identify patterns that aggregate statistics alone do not reveal.

### 3. Validate Potential Thresholds Before Operational Use

The current analysis does not establish a universal network-performance threshold associated with manufacturing efficiency. Any proposed threshold should be validated against additional data and the requirements of the specific manufacturing environment.

### 4. Improve Timestamp Reliability

Investigate the unusual timestamp pattern identified on 1 March 2025 before relying on fine-grained temporal comparisons. Reliable timestamps are important when evaluating the sequence and timing of operational events.

### 5. Extend the Analysis with Additional Evidence

Future investigations could use longer observation periods, independently verified network measurements, and additional operational context to assess whether the observed relationships persist across machines and conditions.

**Practical implication:** This project provides an exploratory framework for examining network and manufacturing indicators together. Its findings can guide further investigation, but they are not sufficient on their own to justify operational changes or establish the causal impact of network conditions.

---

## Limitations and Future Scope

### Limitations

Although the project provides an interactive framework for exploring network and manufacturing indicators, several limitations should be considered when interpreting its results.

- **Observational analysis:** The analysis identifies descriptive patterns and statistical associations; it does not establish causal relationships between network conditions and manufacturing outcomes.
- **Dataset scope:** The findings are based on the available observations from 1 January to 10 March 2025 across 50 machines. They may not generalize to other manufacturing environments or longer operating periods.
- **6G verification:** The analysis does not independently verify that the dataset was collected during a real-world 6G deployment or controlled 6G experiment.
- **Timestamp anomaly:** The unusual timestamp pattern observed on 1 March 2025 introduces an additional consideration for time-based analysis.
- **Metric design:** The Network Stability Index (NSI) is a project-defined metric with equal weighting of normalized latency and packet loss. Different normalization methods or weights could produce different values.
- **Unobserved factors:** Other operational factors may influence production and quality outcomes. The available analysis cannot rule out their contribution to the observed patterns.
- **Descriptive comparisons:** Group comparisons and fitted slopes summarize the available data but should not be treated as causal estimates or guarantees of future performance.

### Future Scope

The project could be extended through the following improvements:

1. **Longer observation periods:** Analyze additional time periods to investigate whether the observed patterns persist over time.
2. **More reliable temporal validation:** Investigate timestamp anomalies and verify the consistency of time-based records before conducting finer-grained temporal analysis.
3. **Additional operational variables:** Incorporate further machine, production, and network measurements where reliable data is available.
4. **Machine-level analysis:** Investigate whether relationships differ across machines after accounting for relevant operating conditions.
5. **Advanced statistical analysis:** Explore multivariable models and appropriate statistical tests to examine relationships while accounting for additional observed factors.
6. **Validated predictive modelling:** If sufficient representative data becomes available, evaluate predictive models using suitable validation procedures and performance metrics.
7. **Deployment and monitoring:** Explore how the dashboard could support ongoing monitoring using validated data pipelines and clearly defined operational requirements.

---

## Local Setup and Usage

Follow these steps to run the dashboard locally.

### Prerequisites

- Python 3.9 or later
- Git
- pip (Python package installer)

### 1. Clone the Repository

```bash
git clone https://github.com/ikalra858-cell/6g-manufacturing-analytics.git
cd 6g-manufacturing-analytics
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r dashboard/requirements.txt
```

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to explore the dashboard.

### Notes

- Ensure that the repository's expected data and output paths are available when running the application.
- If you encounter a missing-package error, verify that the virtual environment is activated and that the required dependencies are installed.

---

## Live Demo

Explore the deployed interactive dashboard:

**[Launch the 6G Manufacturing Analytics Dashboard](https://6g-manufacturing-analytics-uwwmlrkw9p7lbgzmjpwmrk.streamlit.app/)**

The live application allows users to explore network-performance indicators, manufacturing efficiency, production speed, quality metrics, and operation modes through interactive visualizations and filters.

**Source code:** [GitHub Repository](https://github.com/ikalra858-cell/6g-manufacturing-analytics)

---

## Internship Context

This project was developed as part of my **Machine Learning Internship at Unified Mentor Private Limited**.

The project provided practical experience in:

- Working with structured manufacturing data.
- Conducting exploratory data analysis and statistical association analysis.
- Investigating relationships between network indicators and manufacturing outcomes.
- Building an interactive dashboard with Streamlit and Plotly.
- Communicating analytical findings while acknowledging data-quality concerns and methodological limitations.

The project emphasizes evidence-based analysis and responsible interpretation rather than unsupported causal claims.

---

## Conclusion

This project presents an exploratory approach to investigating network performance and manufacturing efficiency in a smart-factory context.

Using statistical comparisons, visual analysis, and an interactive dashboard, it examines network latency and packet loss alongside production speed, quality indicators, efficiency classifications, and machine-condition variables.

Within the available dataset, the analysis found little measurable monotonic association between the selected network indicators and manufacturing outcomes. These results demonstrate the importance of validating assumptions with data and considering operational context before drawing conclusions.

The project provides a foundation for further investigation using additional validated observations, improved temporal consistency, and more advanced analytical methods.

---

## Author

**Isha Kalra**  
BCA Third-Year Student  
Dewan V.S. Group of Institutions, Meerut, India

**GitHub:** [ikalra858-cell](https://github.com/ikalra858-cell)

**Project Repository:** [6G Manufacturing Analytics](https://github.com/ikalra858-cell/6g-manufacturing-analytics)

