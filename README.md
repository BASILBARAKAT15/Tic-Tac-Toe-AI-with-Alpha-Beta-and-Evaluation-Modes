# Tic-Tac-Toe AI with Alpha-Beta Pruning and Evaluation Modes



---

## 🚀 Overview

This project implements a **smart Tic-Tac-Toe AI** that can play against a human player using:

- **Alpha-Beta pruning** for efficient move search
- Two evaluation modes:
  - **Classical heuristic evaluation** (hand-coded)
  - **Machine-learned evaluation** using a trained ML model

The game provides a **professional GUI** with **responsive design**, allowing users to choose difficulty, evaluation method, and player side.

---

## 🎯 Features

- **AI Opponent**:
  - Difficulty levels: **Easy**, **Normal**, **Hard**
  - Option to select **Heuristic** or **ML evaluation**
  - Displays AI move evaluation scores in real-time
- **User Interaction**:
  - Choose **X or O**
  - **Responsive and professional GUI**
  - Color-coded board and options
- **ML Model**:
  - RandomForestRegressor trained on Tic-Tac-Toe dataset
  - Features: number of X/O marks, potential winning lines, center/corner control

---

## 🛠 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/BASILBARAKAT15/Tic-Tac-Toe-AI-with-Alpha-Beta-and-Evaluation-Modes.git
cd Tic-Tac-Toe-AI-with-Alpha-Beta-and-Evaluation-Modes
Install dependencies:

bash
نسخ الكود
pip install numpy pandas scikit-learn customtkinter
Add the dataset:

Ensure tictactoe_dataset.csv is in the project folder (used for ML evaluation).

🎮 How to Play
Run the game:
bash
نسخ الكود
python test.py
Use the Options Panel to:

Choose your side (X or O)

Select difficulty level

Select evaluation function (Heuristic or ML)

Click Start Game

Play against the AI and see evaluation scores on each available move.

🧠 AI Mechanics
Alpha-Beta Pruning:

Searches for optimal moves while pruning unnecessary branches

Depth varies by difficulty (Easy → random moves, Normal → depth 2, Hard → full depth)

Heuristic Evaluation:

Scores board states based on potential winning lines, center/corner occupancy

Machine-Learned Evaluation:

RandomForestRegressor predicts move quality

Features:

Number of X and O marks

Lines close to winning

Center and corner occupancy

Label: +1 if X wins, -1 if O wins

🎨 GUI & UX
Fully responsive frames and buttons

Color-coded X, O, and highlighted winning lines

Dynamic evaluation scores displayed on each cell

Modern, clean design with professional fonts and panels

📸 Screenshots

Option panel for selecting difficulty, side, and evaluation mode.


Gameplay with evaluation scores displayed.

📂 Project Structure
bash
نسخ الكود
AIProject/
│
├─ test.py               # Main game code with GUI and AI
├─ tictactoe_dataset.csv # Dataset for ML evaluation
└─ README.md             # Project documentation
⚡ License
This project is open-source under the MIT License.
