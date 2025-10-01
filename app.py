import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance


# env = dotenv_values(".env")
# ## Secrets using Streamlit Cloud Mechanism
# #https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
# if 'QDRANT_URL' in st.secrets:
#     env['QDRANT_URL'] = st.secrets['QDRANT_URL']
# if 'QDRANT_API_KEY' in st.secrets:
#     env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']
# ###


# @st.cache_resource
# def get_qdrant_client():
#     client = QdrantClient(
#         url=env["QDRANT_URL"],
#         api_key=env["QDRANT_API_KEY"],
    # )

# Tytuł z kolorami i ramką
st.markdown(
    """
    <div style="
        display: flex;
        justify-content: center;
        border: 2px solid #4DFFBE;
        background-color: #FF2DD1;
        padding: 10px;
        border-radius: 15px;
        width: 100%;
        box-sizing: border-box;
        opacity: 0;
        animation: fadeIn 7s forwards;
        animation-delay: 1s;
        font-family: Arial, sans-serif;
    ">
        <h1 style="color: #4DFFBE; margin: 0; text-align: center;">Witaj Przyjacielu</h1>
    </div>
    
    <style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# Zmienna potrzebna do wczytania modelu z punktu 45
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

# Wczytywanie danych
DATA = 'welcome_survey_simple_v2.csv'

# Zmienna potrzebna do wczytania modelu z punktu 50
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

# Funkcja ładuje model
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

# Funkcja ładuje nazwy i opisy klastrów
@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())


# Funkcja odczytuje dane z pliku CSV, klasteryzuje je i zapisuje w pamięci podręcznej
@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    
    # Uwaga: Usunięto drugi, nieosiągalny 'return all_df'
    return df_with_clusters

# Tworzy panel boczny - Formularz do wprowadzenia danych użytkownika
with st.sidebar:
    st.header("Powiedz coś o sobie") # Tworzy nagłówek
    st.markdown("Tutaj masz szansę znaleźć ludzi podobnych do Ciebie!")
    age = st.selectbox("Wiek",['Wybierz', '<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])  # Tworzy pole wyboru Wiek z argumentami
    edu_level = st.selectbox("Wykształcenie", ['Wybierz', 'Podstawowe', 'Średnie', 'Wyższe'])  # Tworzy pole wyboru Wykształcenie z argumentami
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Wybierz', 'Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])  # Tworzy pole wyboru Ulubione zwierzęta z argumentami
    fav_place = st.selectbox("Ulubione miejsce", ['Wybierz', 'Nad wodą', 'W lesie', 'W górach', 'Inne'])  # Tworzy pole wyboru Ulubione miejsce z argumentami
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])  # Tworzy etykietę Płeć z dwoma przyciskami

#  Tworzy DataFrame z danymi wprowadzonymi przez użytkownika
    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])
# Tworzenie zakładek
t1, t2, t3 = st.tabs(["Analiza graficzna", "Analiza statystyczna", "Przegląd danych"])

with t1:

    # Wczytuje model i dane wszystkich uczestników
    model = get_model()
    all_df = get_all_participants()
    cluster_names_and_descriptions = get_cluster_names_and_descriptions()

    # Zabezpieczenie przed błędem, jeśli użytkownik nie wybrał żadnej wartości (selectbox ma 'Wybierz')
    # Zakładam, że pycaret radzi sobie z 'Wybierz', ale najlepiej to sprawdzić w prawdziwym kodzie.
    try:
        predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
        predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

        st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
        st.markdown(predicted_cluster_data['description'])
        same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
        st.metric("Liczba twoich znajomych", len(same_cluster_df))

        st.header("Osoby z grupy")
        
        # Użycie age zamiast fav_animals - Poprawienie błędnej kolumny w oryginalnym kodzie
        # Poprawa nazwy kolumny w jednym z histogramów, które miały age zamiast fav_animals.
        
        # Rozkład wieku
        fig = px.histogram(same_cluster_df, x="age", color_discrete_sequence=["#083C5A"])
        fig.update_layout(
            title="Rozkład wieku w grupie",
            xaxis_title="Wiek",
            yaxis_title="Liczba osób",
        )
        st.plotly_chart(fig)

        # Rozkład wykształcenia
        fig = px.histogram(same_cluster_df, x="edu_level", color_discrete_sequence=["#88BEF5"])
        fig.update_layout(
            title="Rozkład wykształcenia w grupie",
            xaxis_title="Wykształcenie",
            yaxis_title="Liczba osób",
        )
        st.plotly_chart(fig)

        # Rozkład ulubionych zwierząt - POPRAWIONA KOLUMNA x
        fig = px.histogram(same_cluster_df, x="fav_animals", color_discrete_sequence=["#4CB648"])
        fig.update_layout(
            title="Rozkład ulubionych zwierząt w grupie",
            xaxis_title="Ulubione zwierzęta",
            yaxis_title="Liczba osób",
        )
        st.plotly_chart(fig)

        # Rozkład ulubionych miejsc
        fig = px.histogram(same_cluster_df, x="fav_place", color_discrete_sequence=["#BA53DE"])
        fig.update_layout(
            title="Rozkład ulubionych miejsc w grupie",
            xaxis_title="Ulubione miejsce",
            yaxis_title="Liczba osób",
        )
        st.plotly_chart(fig)

        # Rozkład płci
        fig = px.histogram(same_cluster_df, x="gender", color_discrete_sequence=["#FCC72C"])
        fig.update_layout(
            title="Rozkład płci w grupie",
            xaxis_title="Płeć",
            yaxis_title="Liczba osób",
        )
        st.plotly_chart(fig)

    except Exception as e:
        # Możesz wyświetlić ostrzeżenie, jeśli model nie jest w stanie dokonać predykcji
        st.warning(f"Nie można przewidzieć grupy dla podanych danych. Upewnij się, że wybrano wszystkie pola. Błąd: {e}")
        st.stop()


with t2:

    st.header("Statystyki opisowe według grup")

    all_df = get_all_participants()

    # Dla każdej cechy, grupując po niej i wyświetlając statystyki
    grouping_features = ['age', 'edu_level', 'fav_animals', 'fav_place', 'gender']

    for feature in grouping_features:
        st.subheader(f"Statystyki dla {feature}")
        # Grupujemy po danej cesze i liczymy wystąpienia dla klastrów
        grouped = all_df.groupby('Cluster')[feature].value_counts().unstack(fill_value=0)
        st.dataframe(grouped) # Używamy st.dataframe zamiast st.write dla lepszego formatowania

with t3:

    # Wczytanie pliku JSON z dysku (jeśli ten fragment jest tylko do wyświetlenia metadanych klastrów)
    try:
        with open('welcome_survey_cluster_names_and_descriptions_v2.json', 'r', encoding='utf-8') as f:
            clusters_info = json.load(f)
    except FileNotFoundError:
        st.error("Nie znaleziono pliku 'welcome_survey_cluster_names_and_descriptions_v2.json'.")
        st.stop()
        
    # UWAGA: all_df musi być dostępne. Wczytaj je ponownie na wszelki wypadek
    all_df = get_all_participants()

    # --- POPRAWKA DLA WYKRESU SUNBURST ---
    
    # Grupujemy dane, aby uzyskać zliczenia dla każdego poziomu hierarchii (Cluster -> Wiek -> Płeć...)
    # Używamy wszystkich kolumn kategorycznych w celu wygenerowania pełnego podziału Sunburst
    df_sunburst = all_df.groupby(['Cluster', 'age', 'edu_level', 'fav_place', 'gender']).size().reset_index(name='Liczba')

    # Zamiana ID klastrów na ich pełne nazwy dla czytelności na wykresie
    cluster_mapping = {id: details.get('name', id) for id, details in clusters_info.items()}
    df_sunburst['Nazwa Klastra'] = df_sunburst['Cluster'].map(cluster_mapping)

    # Tworzymy wykres Sunburst
    fig = px.sunburst(
        df_sunburst,
        # Używamy Nazw Klastrów jako pierwszego poziomu, a następnie cech
        path=['Nazwa Klastra', 'age', 'edu_level', 'fav_place', 'gender'],
        # Używamy nowo utworzonej kolumny 'Liczba' jako wartości
        values='Liczba', 
        color='Cluster',  # Przypisujemy kolory do oryginalnych ID klastrów
        color_discrete_map={
            "Cluster 0": "red",
            "Cluster 1": "blue",
            "Cluster 2": "green",
            "Cluster 3": "orange",
            "Cluster 4": "purple",
            "Cluster 5": "pink",
            "Cluster 6": "cyan",
            "Cluster 7": "magenta",
            "(?)": "gray" # Dodanie koloru dla nieznanych klastrów
        },
        title="Rozkład cech w klastrach (Sunburst Chart)"
    )

    # Powiększenie wykresu
    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        height=700, # Zmniejszenie, aby lepiej pasowało do ekranu
    )

    # Wyświetlenie w Streamlit
    st.plotly_chart(fig)

    # --- Wyświetlenie metadanych klastrów ---

    # Wyświetlenie każdego klastra jako osobnej tabeli
    st.header("Opis klastrów")
    for cluster_id, data in clusters_info.items():
        st.subheader(f"Klaster: {cluster_id} - {data.get('name', '')}")
        
        # Tworzymy słownik z danymi, które chcemy wyświetlić w tabeli
        table_data = {
            'Nazwa': data.get('name', ''),
            'Opis': data.get('description', '')
        }
        
        # Konwersja na DataFrame i wyświetlenie
        df = pd.DataFrame([table_data])
        st.table(df)