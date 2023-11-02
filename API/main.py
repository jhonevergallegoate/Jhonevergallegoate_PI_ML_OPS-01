# Autor: Jhon Ever Gallego
# Descripción: API para el proyecto de Machine Learning Ops

# Importamos las librerías necesarias.
import pandas as pd
import numpy as np
import gzip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi import HTTPException

app = FastAPI()

# Raiz.
@app.get("/")
def Hello():
    return 'Author: Jhon Ever Gallego Atehortua'

# Creamos la ruta de los datos.
df_games_items_path = './df_games_items.csv'
df_games_path = './df_games.csv'
df_steam_path = './df_steam.csv.gz'

# Cargamos los datos.
df_games_items = pd.read_csv(df_games_items_path)
df_games = pd.read_csv(df_games_path)
data = pd.read_csv(df_steam_path, compression="gzip")

'''
Función 1:  def PlayTimeGenre(genero : str): Debe devolver año con mas horas jugadas para dicho género.
                    Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}
'''

# LOCAL HOST: http://localhost:8000/PlayTimeGenre/action
# REQUEST URL: https://jhonevergallegoate-pi-ml-ops-01.onrender.com/PlayTimeGenre/action
@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre(genero: str):
    try:
        # Filtrar el DataFrame por el género especificado
        df_genre = df_games_items[df_games_items['genres'].str.contains(genero, case=False)]  # Usar case=False para hacerlo insensible a mayúsculas/minúsculas

        if df_genre.empty:
            return {"message": f"No hay datos de horas jugadas para el género {genero}"}
        # Hacer una copia del DataFrame filtrado
        df_genre = df_genre.copy()
        # Convierte la columna 'release_date' en tipo datetime
        df_genre['release_date'] = pd.to_datetime(df_genre['release_date'], format='%Y-%m-%d', errors='coerce')
        # Extraer el año de la columna 'release_date'
        df_genre['release_year'] = df_genre['release_date'].dt.year
        # Agrupar por año, calcular las horas jugadas totales
        result = df_genre.groupby(df_genre['release_year'])['playtime_forever'].sum().reset_index()

        if result.empty:
            return {"message": f"No hay datos de horas jugadas para el género {genero}"}
        # Encontrar el año con más horas jugadas para el género
        max_year = result.loc[result['playtime_forever'].idxmax()]

        return {"Año de lanzamiento con más horas jugadas para Género " + genero: int(max_year['release_year'])}

    except Exception as e:
        return {"error": str(e)}

'''
Función 2:  def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista
            de la acumulación de horas jugadas por año.
                    Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, 
                        "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}
'''

# LOCAL HOST: http://localhost:8000/UserForGenre/action
# REQUEST URL: https://jhonevergallegoate-pi-ml-ops-01.onrender.com/UserForGenre/action
@app.get("/UserForGenre/{genero}")
def UserForGenre(genero: str):
    try:
        # Filtrar el DataFrame por el género especificado
        df_genre = df_games_items[df_games_items['genres'].str.contains(genero, case=False)]  # Usar case=False para hacerlo insensible a mayúsculas/minúsculas
        if df_genre.empty:
            return {
                f"Usuario con más horas jugadas para Género {genero}": None,
                "Horas jugadas": []
            }

        # Hacer una copia del DataFrame filtrado
        df_genre = df_genre.copy()
        # Convierte la columna 'release_date' en tipo datetime
        df_genre['release_date'] = pd.to_datetime(df_genre['release_date'], format='%Y-%m-%d', errors='coerce')
        # Extraer el año de la columna 'release_date'
        df_genre['release_year'] = df_genre['release_date'].dt.year
        # Agrupar por usuario y año, calcular las horas jugadas totales
        result = df_genre.groupby(['user_id', df_genre['release_year']])['playtime_forever'].sum().reset_index()

        if result.empty:
            return {
                f"No hay datos de horas jugadas para el género {genero}": None,
                "Horas jugadas": []
            }

        # Encontrar al usuario con más horas jugadas para el género
        max_user = result.loc[result.groupby('user_id')['playtime_forever'].idxmax()]
        max_user_id = max_user['user_id'].values[0]
        # Calcular la acumulación de horas jugadas por año
        total_hours_by_year = result.groupby('release_year')['playtime_forever'].sum().reset_index()
        # Crear la lista de acumulación de horas jugadas por año en el formato deseado
        horas_por_ano = [{"Año": int(row['release_year']), "Horas": int(row['playtime_forever'])} for _, row in total_hours_by_year.iterrows()]

        return {
            f"Usuario con más horas jugadas para Género {genero}": max_user_id,
            "Horas jugadas": horas_por_ano
        }

    except Exception as e:
        return {"error": str(e)}  # Devolvemos un mensaje de error en caso de excepción

'''
Función 3:  def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. 
                (reviews.recommend = True y comentarios positivos/neutrales)
            Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
'''

# LOCAL HOST: http://localhost:8000/UsersRecommend/2011
# REQUEST URL: https://jhonevergallegoate-pi-ml-ops-01.onrender.com/UsersRecommend/2011
@app.get("/UsersRecommend/{year}")
def UsersRecommend(year: int):
    try:
        # Filtramos por año
        data_year = data[data["release_date"] == year]
        # Filtramos por recomendados
        data_year = data_year[data_year["recommend"] == True]
        # Filtramos por comentarios positivos/neutrales
        data_year = data_year[data_year["sentiment"] > 0]
        # Agrupamos por juego y contamos las recomendaciones
        data_year = data_year.groupby("app_name")["recommend"].count().reset_index()
        # Ordenamos de mayor a menor
        data_year = data_year.sort_values(by="recommend", ascending=False)
        # Obtenemos el top 3
        top3 = data_year.head(3)

        # Formateamos el resultado en el formato deseado
        result = [{"Puesto 1": top3.iloc[0]["app_name"]},
                {"Puesto 2": top3.iloc[1]["app_name"]},
                {"Puesto 3": top3.iloc[2]["app_name"]}]

        return result

    except KeyError:
        raise HTTPException(status_code=404, detail="No se encontraron datos para el año especificado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error inesperado: {str(e)}")

'''
Función 4:  def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.
                (reviews.recommend = False y comentarios negativos)
            Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
'''

# LOCAL HOST: http://localhost:8000/UsersNotRecommend/2009
# REQUEST URL: https://jhonevergallegoate-pi-ml-ops-01.onrender.com/UsersNotRecommend/2009
@app.get("/UsersNotRecommend/{year}")
def UsersNotRecommend(year: int):
    try:
        # Filtramos por año
        data_year = data[data["release_date"] == year]
        # Filtramos por no recomendados
        data_year = data_year[data_year["recommend"] == False]
        # Filtramos por comentarios negativos
        data_year = data_year[data_year["sentiment"] == 0]
        # Agrupamos por juego y contamos las recomendaciones
        data_year = data_year.groupby("app_name")["recommend"].count().reset_index()
        # Ordenamos de mayor a menor
        data_year = data_year.sort_values(by="recommend", ascending=False)
        # Obtenemos el top 3
        top3 = data_year.head(3)

        # Formateamos el resultado en el formato deseado
        result = [{"Puesto 1": top3.iloc[0]["app_name"]},
                {"Puesto 2": top3.iloc[1]["app_name"]},
                {"Puesto 3": top3.iloc[2]["app_name"]}]

        return result

    except KeyError:
        raise HTTPException(status_code=404, detail="No se encontraron datos para el año especificado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error inesperado: {str(e)}")

'''
Función 5:  def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros
                de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
            Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}
'''

# LOCAL HOST: http://localhost:8000/sentiment_analysis/2010
# REQUEST URL: https://jhonevergallegoate-pi-ml-ops-01.onrender.com/Sentiment_Analysis/2010
@app.get("/Sentiment_Analysis/")
def Sentiment_Analysis(year: int):
    # Filtramos por año
    data_year = data[data["release_date"] == year]
    # Agrupamos por sentimiento y contamos las reseñas
    data_year = data_year.groupby("sentiment")["review"].count().reset_index()
    # Obtenemos el top 3
    sentiment = data_year.to_dict("records")
    # Inicializar contadores
    negative_count = 0
    neutral_count = 0
    positive_count = 0
    # Contar el número de reseñas con cada sentimiento
    for s in sentiment:
        if s["sentiment"] == 0:
            negative_count += s["review"]
        elif s["sentiment"] == 1:
            neutral_count += s["review"]
        elif s["sentiment"] == 2:
            positive_count += s["review"]
    # Crear el diccionario con los contadores
    sentiment = {
        "Negative": negative_count,
        "Neutral": neutral_count,
        "Positive": positive_count,

    }
    return {"Según el año de lanzamiento": year, "Sentimiento": sentiment}

'''
Sistema de recomendación:  def recommendation_system( usuario : str ): Recibe un usuario y devuelve una lista con los 5 juegos
'''

# LOCAL HOST: http://localhost:8000/recomendacion_juego/6210
# REQUEST URL: https://jhonevergallegoate-pi-ml-ops-01.onrender.com/Recomendacion_Juego/6210
@app.get("/Recomendacion_Juego/{product_id}")
async def Recomendacion_Juego(product_id:int):
    try: 
        # Obtiene el juego de referencia
        target_game = df_games[df_games["id"] == product_id]
        if target_game.empty:
            return {"message": "No se encontró el juego de referencia"}
        # Combina las etiquetas tags y genres en una sola cadena de texto
        target_game_tags_and_genres = " ".join(target_game["tags"].fillna(" ").astype(str) + " " + target_game["genres"].fillna(" ").astype(str))

        # Crea un vectorizador TF-IDF
        tfidf_vectorizer = TfidfVectorizer()

        # Configura el tamaño del lote para la lectura del juego
        chunk_size = 100  # Tamaño del lote
        similarity_scores = None

        # Procesa los juegos por lotes aplicando chunks
        for chunk in pd.read_csv(df_games_path, chunksize=chunk_size):
            # Combina las etiquetas tags y genres en una sola cadena de texto
            chunk_tags_and_genres = " ".join(chunk["tags"].fillna(" ").astype(str) + " " + chunk["genres"].fillna(" ").astype(str))

            # Aplica el vectorizador TF-IDF al lote actual de juegos y al juego de referencia
            tfidf_matrix = tfidf_vectorizer.fit_transform([target_game_tags_and_genres, chunk_tags_and_genres])

            # Calcula la similitud entre el juego de referencia y los juegos del lote actual
            similarity_scores_batch = cosine_similarity(tfidf_matrix)

            if similarity_scores is None:
                similarity_scores = similarity_scores_batch
            else:
                similarity_scores = np.concatenate((similarity_scores, similarity_scores_batch), axis=1)

        if similarity_scores is not None:
            # Obtiene los índices de los juegos similares 
            similar_games_indices = similarity_scores[0].argsort()[::-1]
            # Recomienda los juegos más similares 
            num_recomendation = 5
            recommended_games = df_games.iloc[similar_games_indices[1:num_recomendation + 1]]
            # Devuelve la lista con los juegos recomendados 

            return recommended_games[["app_name", "id"]].to_dict(orient="records")

        return {"message": "No se encontraron juegos similares"}
    except Exception as e:
        return {"Message": f"Error: {str(e)}"}