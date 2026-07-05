# Modelling Urban Travel Decisions with Machine Learning

A machine learning-based system for predicting and personalising urban travel mode choices using trip characteristics, weather conditions, behavioural data, and user preferences.

This project was developed as part of my MSc Data Science final project. It explores how machine learning can support more user-centred urban mobility recommendations by predicting likely travel modes and adapting the final recommendation based on individual priorities such as vehicle availability, eco-friendliness, and time sensitivity.

---

## Project Overview

Most travel applications focus on route optimisation, but travel mode choice is influenced by many contextual and personal factors, including trip distance, travel duration, trip purpose, weather conditions, vehicle availability, and user priorities.

This project addresses that gap by building a two-stage recommendation system:

1. **Machine Learning Prediction Layer**  
   Predicts the most likely travel mode based on trip-level, temporal, spatial, and weather-related features.

2. **Personalised Recommendation Layer**  
   Adjusts the model output using user preferences, such as whether the user owns a car or bicycle, whether they prefer eco-friendly options, and whether they are time-sensitive.

The final output is displayed through an interactive web dashboard.

---

## Problem Statement

Urban travel decisions are rarely based on one factor alone. A model may predict the most common transport mode for a journey, but that does not always mean it is the most suitable option for a specific user.

For example, two people making the same trip may have different priorities:

- One may prioritise speed.
- Another may prefer a lower-emission option.
- Another may not have access to a car or bicycle.
- Another may need a more accessible or practical travel mode.

The challenge was therefore to move beyond simple mode prediction and build a system that combines predictive modelling with personalised decision support.

---

## Key Features

- Predicts five urban travel modes:
  - Walking
  - Bicycle
  - Car driver
  - Car passenger
  - Public bus

- Uses real-world travel survey data from the Queensland Household Travel Survey.

- Integrates weather-related variables such as:
  - Temperature
  - Wind speed
  - Precipitation

- Applies a Random Forest classifier to model travel behaviour.

- Includes a personalised utility-based layer that adjusts recommendations according to:
  - Car availability
  - Bicycle availability
  - Eco-friendly preference
  - Time sensitivity

- Provides an interactive web dashboard with:
  - Trip simulator controls
  - Predicted travel mode
  - Prediction confidence
  - Key factors influencing the decision
  - CO₂ emissions comparison
  - Accessibility considerations

---

## Dataset

The project uses the **Queensland Household Travel Survey 2021–2024** dataset, which contains trip-level information about urban travel behaviour.

The dataset was processed and combined with historical weather data to create a richer modelling dataset.

Main feature groups include:

| Feature Group | Examples |
|---|---|
| Trip characteristics | Trip distance, trip duration, trip purpose |
| Temporal variables | Departure time, day, month, year |
| Weather variables | Temperature, wind speed, precipitation |
| Spatial variables | Origin and destination coordinates |

---

## Methodology

The project followed a structured applied data science workflow:

### 1. Data Preprocessing

The raw travel survey data was cleaned and filtered to focus on the selected travel modes. Weather data was added using an API to match trip conditions with environmental variables.

### 2. Exploratory Data Analysis

Exploratory analysis was used to understand travel mode distributions, detect class imbalance, and identify relationships between travel modes, distance, and duration.

One key challenge was class imbalance. Car-based modes appeared much more frequently than bicycle and public bus trips, which created a risk that the model would favour majority classes.

### 3. Feature Engineering

The model used trip, temporal, spatial, and weather-related features. Categorical variables such as trip purpose were encoded for machine learning, while numerical variables were kept in continuous form.

### 4. Model Development

Several modelling options were considered, with a focus on accuracy, interpretability, and suitability for a dashboard-based system.

The final model used was a **Random Forest Classifier**, selected because it can handle non-linear relationships, manage complex feature interactions, and provide feature importance values for interpretability.

### 5. Personalisation Layer

The personalisation layer was added on top of the model prediction.

Instead of treating the predicted transport mode as the final answer, the system adjusts the recommendation using a utility-based approach. This allows the system to reflect user constraints and preferences.

For example:

- If the user does not own a car, car-based options are penalised.
- If the user selects eco-friendly preferences, walking, cycling, and public bus become more favourable.
- If the user is time-sensitive, faster modes receive a higher score.
- If walking or cycling is impractical because of distance or weather, those modes can be reduced.

This makes the recommendation more practical and user-centred.

---

## Model Performance

The Random Forest model achieved strong performance on the test dataset.

| Metric | Result |
|---|---|
| Accuracy | 0.81 |
| Weighted F1-score | 0.79 |
| Best-performing common mode | Car driver |
| Strong active-mode performance | Walking |
| Main limitation | Lower recall for minority classes such as bicycle and public bus |

The model performed best on common travel modes, especially car driver and walking. Performance was weaker for underrepresented classes such as bicycle and public bus due to class imbalance in the dataset.

---

## Dashboard

The dashboard allows users to simulate a trip and receive a personalised travel mode recommendation.

Users can input:

- Date and time
- Current location
- Destination
- Trip purpose
- Car availability
- Bicycle availability
- Eco-friendly preference
- Time sensitivity

The dashboard then displays:

- Recommended travel mode
- Prediction confidence
- Key influencing factors
- Estimated CO₂ emissions
- Accessibility considerations

---

## Tech Stack

| Area | Tools |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, Random Forest |
| Visualisation | Matplotlib, Seaborn |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data source | Queensland Household Travel Survey |
| Weather data | Open-Meteo API |

---

## Project Structure

```text
Modelling-Urban-Travel-Decisions-with-Machine-Learning/
│
├── api.py                         # Flask backend, model training, prediction API
├── index.html                     # Dashboard interface
├── style.css                      # Dashboard styling
├── trips_with_coordinates.xlsx    # Processed travel dataset
├── report.pdf                     # Full MSc project report
└── README.md                      # Project documentation
