# CO2 Emissions App 🌍

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)

## Description

This Streamlit application provides a CO2 Emission Dashboard and a machine learning-based prediction tool. It allows users to visualize CO2 emission data and predict CO2 emissions based on vehicle specifications. The app offers interactive plots and data exploration features to understand the factors influencing CO2 emissions. It uses a RandomForestRegressor model trained on a dataset of vehicle specifications and their corresponding CO2 emissions.

## Table of Contents

1.  [Features](#features)
2.  [Tech Stack](#tech-stack)
3.  [Installation](#installation)
4.  [Usage](#usage)
5.  [Project Structure](#project-structure)
6.  [Contributing](#contributing)
7.  [License](#license)
8.  [Important Links](#important-links)

## Features

*   **Interactive Dashboard:** Visualize CO2 emissions data using interactive plots created with Matplotlib and Seaborn.
*   **Data Exploration:** Explore the dataset using interactive dataframes and plots showing distributions of various features.
*   **CO2 Emission Prediction:** Predict CO2 emissions based on engine size, cylinders, and fuel consumption using a trained machine learning model.
*   **Outlier Handling:** The application handles outliers by removing them from the dataset before training the model, leading to more robust predictions.
*   **Input Validation:** Includes input validation to ensure realistic vehicle specifications are used for predictions.
*   **Background Animation:** The user interface has a background animation to improve the aesthetic appeal of the dashboard.

## Tech Stack

*   **Python:** Primary programming language.
*   **Streamlit:** Web framework for creating interactive dashboards.
*   **Pandas:** Data manipulation and analysis.
*   **Scikit-learn:** Machine learning library for model training and prediction.
*   **Matplotlib & Seaborn:** Data visualization.
*   **Numpy & Scipy**: Data manipulation and scientific computing

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Adityaphophale/co2-emissions-app.git
    cd co2-emissions-app
    ```
2.  Install the required dependencies. Create a virtual environment to prevent conflicts:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux and macOS
    venv\Scripts\activate  # On Windows
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the Streamlit application:
    ```bash
    streamlit run app\(2\).py
    ```
2.  Open the application in your web browser. The app will usually open automatically, but if not, it will provide a local URL in the terminal that you can use.
3.  Navigate the dashboard to explore CO2 emission visualizations or use the prediction tool by selecting "Model Prediction" from the sidebar.
4.  If you selected "Model Prediction", enter the vehicle specifications (Engine Size, Cylinders, Fuel Consumption) in the sidebar and click "Predict Emissions". The predicted CO2 emissions will be displayed.

**Use Cases:**

*   **Educational Tool**: Can be used as an educational tool to understand the relationship between car engine specifications and their impact on CO2 emissions.
*   **Policy Making**: Policy makers can use the tool to set realistic emission targets by visualizing real-world data.
*   **Vehicle Purchase**: Potential vehicle purchasers can use the prediction feature to understand the environmental impact of different car models before making a purchase.

## Project Structure

```
co2-emissions-app/
├── app (2).py         # Main Streamlit application file
├── co2 Emissions.csv  # Dataset for CO2 emissions
└── requirements.txt   # List of Python dependencies
```

## Contributing

Contributions are welcome! Here are the steps to contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Push your changes to your fork.
5.  Submit a pull request.

## License

No license provided. All rights reserved.

## Important Links

*   **Repository:** [https://github.com/Adityaphophale/co2-emissions-app](https://github.com/Adityaphophale/co2-emissions-app)
--- 

<p align="center">
  <a href="https://github.com/Adityaphophale/co2-emissions-app">CO2 Emissions App</a> • Developed by Aditya Rahul Phophale •  Fork, like, give star and contribute 
</p>
