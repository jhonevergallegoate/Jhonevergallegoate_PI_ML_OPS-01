import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

app = FastAPI()

# Raiz
@app.get("/")
def Hello():
    return 'Author: Jhon Ever Gallego'

df_games_items = pd.read_csv('df_games_items.csv')
df_games_reviews = pd.read_csv('df_games_reviews.csv')
df_games_reviews_false = pd.read_csv('df_games_reviews_false.csv')
df_sentimental = pd.read_csv('df_sentimental.csv')
df_games = pd.read_csv('df_games.csv')
df_games_csv = pd.read_csv('df_games.csv')

'''
Función 1:  def PlayTimeGenre(genero : str): Debe devolver año con mas horas jugadas para dicho género.
                    Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}
'''

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

@app.get("/UsersRecommend/{año}")
def UsersRecommend(año: int):
    try:
        # Filtrar el DataFrame por el año especificado y reseñas recomendadas (recommend = True)
        reviews_filtered = df_games_reviews[(df_games_reviews['posted'].str[:4] == str(año)) & (df_games_reviews['recommend'] == True)]
        if reviews_filtered.empty:
            return {"message": "No hay reseñas recomendadas para el año especificado"}
        # Contar las reseñas de cada juego
        game_counts = reviews_filtered['app_name'].value_counts().reset_index()
        game_counts.columns = ['Juego', 'Total Reseñas']
        # Ordenar los juegos por la cantidad de reseñas en orden descendente
        game_counts = game_counts.sort_values(by='Total Reseñas', ascending=False)
        # Tomar los 3 juegos más recomendados
        top_3_games = game_counts.head(3)
        # Crear la lista de retorno en el formato deseado
        result = [{"Puesto " + str(i + 1): game} for i, game in enumerate(top_3_games['Juego'])]
        return result
    except Exception as e:
        return {"error": str(e)}

'''
Función 4:  def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.
                (reviews.recommend = False y comentarios negativos)
            Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
'''

@app.get("/UsersNotRecommend/{año}")
def UsersNotRecommend(año: int):
    try:
        # Filtrar el DataFrame por el año especificado y reseñas no recomendadas (recommend = False)
        reviews_filtered = df_games_reviews_false[(df_games_reviews_false['posted'].str[:4] == str(año)) & (df_games_reviews_false['recommend'] == False)]
        if reviews_filtered.empty:
            return {"message": "No hay reseñas no recomendadas para el año especificado"}
        # Contar las reseñas de cada juego
        game_counts = reviews_filtered['app_name'].value_counts().reset_index()
        game_counts.columns = ['Juego', 'Total Reseñas']
        # Ordenar los juegos por la cantidad de reseñas en orden descendente
        game_counts = game_counts.sort_values(by='Total Reseñas', ascending=False)
        # Tomar los 3 juegos menos recomendados
        top_3_games = game_counts.head(3)
        # Crear la lista de retorno en el formato deseado
        result = [{"Puesto " + str(i + 1): game} for i, game in enumerate(top_3_games['Juego'])]
        return result
    except Exception as e:
        return {"error": str(e)}
    
'''
Función 5:  def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros
                de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
            Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}
'''

@app.get("/sentiment_analysis/{año}")
def sentiment_analysis(año: int):
    try:
        # Filtrar el DataFrame por el año especificado
        df_year = df_sentimental[df_sentimental['year'] == año]
        if df_year.empty:
            return {"message": "No hay datos de análisis de sentimiento para el año especificado"}
        # Contar los registros de cada categoría de análisis de sentimiento
        result = df_year['sentiment'].value_counts().reset_index()
        result.columns = ['sentiment', 'count']
        # Crear la lista de retorno en el formato deseado
        result = {row['sentiment']: int(row['count']) for _, row in result.iterrows()}
        return result
    except Exception as e:
        return {"error": str(e)}

'''
Sistema de recomendación:  def recommendation_system( usuario : str ): Recibe un usuario y devuelve una lista con los 5 juegos
'''

@app.get("/recomendacion_juego/{product_id}")
async def recomendacion_juego(product_id:int):
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
        for chunk in pd.read_csv("./df_games.csv", chunksize=chunk_size):
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
            recommended_games = df_games_csv.iloc[similar_games_indices[1:num_recomendation + 1]]
            # Devuelve la lista con los juegos recomendados 
            return recommended_games[["app_name", "id"]].to_dict(orient="records")

        return {"message": "No se encontraron juegos similares"}
    except Exception as e:
        return {"Message": f"Error: {str(e)}"}