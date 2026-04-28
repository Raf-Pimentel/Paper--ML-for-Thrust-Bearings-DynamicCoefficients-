# Machine Learning Surrogate Modeling for Thrust Bearings Dynamic Coefficients

This repository contains the source code, data visualizations, and interactive dashboard developed for an undergraduate research project (Iniciação Científica) conducted at the LAMAR laboratory (Unicamp) in collaboration with Prof. Thales Peixoto. 

The primary goal of this codebase is to evaluate and predict the dynamic coefficients (stiffness and damping) of hydrodynamic thrust bearings using Machine Learning regression techniques. This work directly supports the research paper prepared for presentation at the **MECSOL 2026** congress.

## 🚀 Project Overview

Hydrodynamic lubrication relies on complex fluid dynamics (such as the Reynolds equation) to determine pressure distribution, load capacity, and dynamic coefficients in bearings. Solving these equations numerically can be computationally expensive. This project introduces a machine learning approach to act as a surrogate model, rapidly predicting these coefficients while maintaining high accuracy.

**Key Components:**
* **Analytical & Numerical Comparisons:** Validation of pressure distributions ($H_0$) and load capacities ($W_z$).
* **Machine Learning Regression:** Models trained to predict Stiffness ($K$) and Damping ($C$) coefficients (`MLRegressionK` and `MLRegressionC`).
* **Interactive Tribology Dashboard:** A built-in web application to interactively visualize bearing geometries and fluid dynamics.

## 📁 Repository Structure

The project has been organized into the following directories:

\`\`\`text
.
├── src/                    # Core Python scripts for ML models and data generation
│   ├── MLRegressionC_*.py  # Damping coefficient regression models
│   ├── MLRegressionK_*.py  # Stiffness coefficient regression models
│   ├── grafico*_ic.py      # Scripts for generating specific analytical/numerical plots
│   └── Plataforma_Tribologia.py 
├── Mancais_Dashboard/      # Interactive dashboard application
│   ├── main.py             # Dashboard entry point
│   ├── pages/              # Dashboard interactive pages (e.g., Geometry)
│   ├── utils/              # Calculation and visualization helper functions
│   └── requirements.txt    # Dashboard dependencies
├── figures/                # Generated visual assets
│   ├── plots/              # Graphs, pressure distributions, and ML regression results
│   └── summary_photos/     # Additional project imagery
├── docs/                   # Academic literature and theoretical reference materials
└── archives/               # Zipped templates (e.g., MECSOL 2026 LaTeX template) and backups
\`\`\`

## 🛠️ Setup and Installation

**1. Clone the repository:**
\`\`\`bash
git clone https://github.com/Raf-Pimentel/Paper--ML-for-Thrust-Bearings-DynamicCoefficients-.git
cd Paper--ML-for-Thrust-Bearings-DynamicCoefficients-
\`\`\`

**2. Install Dashboard Dependencies:**
Navigate to the dashboard directory and install the required Python packages. It is recommended to use a virtual environment.
\`\`\`bash
cd Mancais_Dashboard
pip install -r requirements.txt
\`\`\`

**3. Run the Interactive Dashboard:**
\`\`\`bash
python main.py 
# (Or 'streamlit run main.py' depending on the framework used)
\`\`\`

## 📊 Visualizations & Results

The `figures/plots/` directory contains outputs validating the model, including:
* $W_z$ (Load Capacity) Analytical vs. Numerical comparisons.
* Pressure distribution across the bearing geometry.
* Regression performance graphics for $K$ and $C$ coefficients.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
