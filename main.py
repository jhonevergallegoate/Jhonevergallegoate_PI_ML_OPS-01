import pandas as pd
from fastapi import FastAPI

app = FastAPI()

df_games_items = pd.read_csv('/.DATA API/df_games_items.csv')

# Función 1.
@app.get('UserForGenre/{genero, df}')
def UserForGenre(genero, df):
    # Filtrar el DataFrame por el género especificado
    df_genre = df[df['genres'].str.lower().str.contains(genero)]

    if df_genre.empty:
        return {
            f"Usuario con más horas jugadas para Género {genero}": None,
            "Horas jugadas": []
        }

    # Agrupar por usuario y año, calcular las horas jugadas totales
    result = df_genre.groupby(['user_id', df_genre['year']])['playtime_forever'].sum().reset_index()

    # Encontrar al usuario con más horas jugadas para el género
    max_user = result.loc[result.groupby('user_id')['playtime_forever'].idxmax()]
    max_user_id = max_user['user_id'].values[0]

    # Calcular la acumulación de horas jugadas por año
    total_hours_by_year = result.groupby('year')['playtime_forever'].sum().reset_index()
    
    # Crear la lista de acumulación de horas jugadas por año en el formato deseado
    horas_por_ano = [{"Año": int(row['year']), "Horas": int(row['playtime_forever'])} for _, row in total_hours_by_year.iterrows()]

    return {
        f"Usuario con más horas jugadas para Género {genero}": max_user_id,
        "Horas jugadas": horas_por_ano
    }