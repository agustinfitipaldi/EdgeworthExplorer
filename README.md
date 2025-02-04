# Edgeworth Box Explorer

An interactive visualization tool for exploring the Edgeworth box diagram, indifference curves, contract curves, and Walrasian equilibrium points.

## Features

- Interactive input of utility functions for both agents
- Adjustable initial endowments
- Real-time visualization of:
  - Indifference curves for both agents
  - Contract curve
  - Walrasian equilibrium points
  - Initial endowment point
- 3D visualization of utility surfaces
- Hover tooltips with detailed information

## Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run main.py
```

## Deployment

### Streamlit Cloud (Recommended)

1. Fork this repository
2. Create an account at [share.streamlit.io](https://share.streamlit.io)
3. Create a new app and select this repository
4. Deploy!

### Alternative: Render.com

1. Create an account at [render.com](https://render.com)
2. Create a new Web Service
3. Connect your repository
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run main.py`
5. Deploy!

## Usage

1. Enter utility functions for both agents using x and y as variables
2. Set initial endowments for both goods (X and Y) for each agent
3. The visualization updates automatically
4. Hover over curves and points to see detailed information

## Example Utility Functions

- Cobb-Douglas: `x^0.5 * y^0.5`
- Perfect Substitutes: `x + y`
- Perfect Complements: `min(x, y)`
- Quasi-linear: `x + ln(y)`
