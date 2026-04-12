#!/bin/bash
# github_setup.sh
# Paleiskite šį skriptą projekto šaknies kataloge

echo "🚀 GitHub projekto sukūrimas..."

git init
git add .
git commit -m "feat: initial project structure - database, models, utils"

git checkout -b develop
echo "# changelog" > CHANGELOG.md
git add CHANGELOG.md
git commit -m "chore: add changelog, switch to develop branch"

git checkout -b feature/neural-network
git add models/neural_network_model.py
git commit -m "feat: add feed forward neural network with 25 experiments"

git checkout develop
git merge feature/neural-network --no-ff -m "merge: neural network feature"

git checkout -b feature/visualizations
git add utils/visualizations.py
git commit -m "feat: add 7 plotly visualizations"

git checkout develop
git merge feature/visualizations --no-ff -m "merge: visualizations feature"

git checkout main
git merge develop --no-ff -m "release: v1.0 - complete churn prediction system"

echo "✅ Git struktūra sukurta!"
echo "Dabar: git remote add origin <jūsų-repo-url> && git push --all"
