from fastapi import FastAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import base64
from io import BytesIO

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Highest-grossing films API is running"}

@app.get("/movies")
def movies():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    tables = pd.read_html(url)
    df = tables[0]

    # Clean money column
    df['Worldwide gross'] = df['Worldwide gross'].replace('[\$,T]', '', regex=True)
    df['Worldwide gross'] = df['Worldwide gross'].replace('T', '', regex=True)
    df['Worldwide gross'] = df['Worldwide gross'].str.replace(',', '', regex=False).astype(float)

    # Q1: Number of $2bn+ movies before 2000
    q1 = df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2000)].shape[0]

    # Q2: Earliest film grossing > $1.5bn
    q2_row = df[df['Worldwide gross'] > 1_500_000_000].sort_values('Year').iloc[0]
    q2 = q2_row['Title']

    # Q3: Correlation between Rank and Peak
    corr = df['Rank'].corr(df['Peak'])
    q3 = round(corr, 3)

    # Q4: Scatter plot with regression
    X = df['Rank'].values.reshape(-1, 1)
    y = df['Peak'].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(5, 4))
    plt.scatter(df['Rank'], df['Peak'], color='blue')
    plt.plot(df['Rank'], y_pred, color='red', linestyle='dotted')
    plt.xlabel("Rank")
    plt.ylabel("Peak")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    img_uri = f"data:image/png;base64,{img_base64}"

    return [q1, q2, q3, img_uri]
