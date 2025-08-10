# TDS Project 2 — Highest Grossing Films API

A FastAPI service deployed on Vercel that scrapes Wikipedia for the list of highest-grossing films and answers:

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between Rank and Peak?
4. Scatterplot of Rank vs. Peak with regression line (base64).

## Endpoints

- `/` — Health check
- `/analyze` — Returns JSON array of answers.

## Deploy to Vercel

```bash
vercel deploy
