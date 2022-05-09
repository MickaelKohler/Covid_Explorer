from calendar import month
from datetime import datetime, date
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from pmdarima import auto_arima


# CONFIG

st.set_page_config(page_title="Covid Explorer",
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="auto",
                   )

st.markdown("""
    <style>
    .titre {
        font-size:16px;
        font-weight:normal;
        margin:0px;
    }
    .text {
        font-size:16px;
        font-weight:normal;
        color:lightgray;
    }
    .sous_indice {
        font-size:60px;
        font-weight:bold;
    }
    .indice_total {
        font-size:100px;
        font-weight:bold;
    }
    </style>
    """, unsafe_allow_html=True)


# FONCTIONS


@st.cache
def load_data(url):
    return pd.read_csv(url)


def load_time_series(url):
    ts = pd.read_csv(url)
    ts['date'] = ts['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    return ts


def space(n):
    '''Créé des espaces de présentation'''
    for _ in range(n):
        st.title(' ')


def evolution_couleur(dataframe):
    '''Prend une time serie d'un pays sous la forme df[['date', 'pays']]
    et retourne les stat d'évolution de l'épidémie'''
    df = dataframe.copy()
    df['evo'] = df.iloc[:, 1].diff()
    df['evo'] = df['evo'].rolling(window=7, center=True).mean()
    df['pourcent'] = round(df['evo'].pct_change()*100, 2)
    df.fillna(0, inplace=True)
    df['color'] = df['pourcent'].apply(lambda x: 'green' if x <=0 else 'red')
    return df


def cas_quotidiens(dataframe):
    '''Prend une time serie d'un pays sous la forme df[['date', 'pays']]
    et retourne les stat quotidienne de l'épidémie'''
    df = dataframe[['date', country]].copy()
    df['evo'] = df.iloc[:, 1].diff()
    df['evo'] = df['evo'].apply(lambda x: max(x, 0))
    df['evo7j'] = df['evo'].rolling(window=7, center=True).mean()
    return df


# DATA

COUNTRIES = "./data/countries.csv"
TS_COUNTRIES = './data/ts_countries.csv'
TS_CONTINENTS = './data/ts_continents.csv'
TS_GLOBAL = './data/ts_glob.csv'
POP_MONDIALE = './data/pop_mondiale.csv'
COUNTRIES_CODE = './data/countries-codes.csv'

df_countries = load_data(COUNTRIES)
ts_countries = load_time_series(TS_COUNTRIES)
ts_continents = load_time_series(TS_CONTINENTS)
ts_global = load_time_series(TS_GLOBAL)
pop_mondiale = load_data(POP_MONDIALE)
countries_code = pd.read_csv(COUNTRIES_CODE, sep=";")


# SIDE BAR

thematiques = [
    'Analyse',
    'Prediction',
    ]

st.sidebar.title('Covid Explorer')
st.sidebar.title(" ")

section = st.sidebar.radio(
    'Selection de la partie : ',
    thematiques)
st.sidebar.subheader(' ')


# MAIN PAGE

if section == 'Analyse':

    st.title("Analyse de l'évolution des contaminations")

    sub_section = st.sidebar.radio(
        'Selection de la sous-section :',
        ['Présentation globale',
         'Comparateur de Continents', 
         'Cartographie',
         'Evolution par Pays',
         'Hot Zone',
        ])
    st.sidebar.title(' ')

    if sub_section == 'Présentation globale':
        st.header("Présentation du dataset")
        with st.sidebar.expander('Options de présentation'):
            limit_range = st.checkbox('Graphiques resserrés')
        space(1)

        st.markdown('''
            Les données des infections du Coronavirus ont été fournies par le **Johns Hopkins Coronavirus Resource Center** 
            et présente **les cas de contamnisation cumulés par jours du 22 janvier 2020 au 10 avril 2021.**
            ''')
        space(1)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                'Nombre de Continents présents',
                len(df_countries['continent_iso'].unique())
                )
        with col2:
            st.metric(
                'Nombre de pays représentés',
                len(df_countries['Country/Region'].unique())
                )
        with col3:
            nb_days = date(2021, 4, 10) - date(2020, 1, 22)
            st.metric(
                'Nombre de jours analysés',
                nb_days.days
                )

        st.markdown('---')

        st.header('Evolution des cas par Continents')

        # Graph TS par pays pour une vision globale
        EU = ts_continents['Europe']
        NA = ts_continents['Europe']+ts_continents['North America']
        AS = ts_continents['Europe']+ts_continents['North America']+ts_continents['Asia']
        SA = ts_continents['Europe']+ts_continents['North America']+ts_continents['Asia']+ts_continents['South America']
        AF = ts_continents['Europe']+ts_continents['North America']+ts_continents['Asia']+ts_continents['South America']+ts_continents['Africa']
        OC = ts_continents['Europe']+ts_continents['North America']+ts_continents['Asia']+ts_continents['South America']+ts_continents['Africa']+ts_continents['Oceania']

        fig_contients = go.Figure([
            go.Scatter(x=ts_continents['date'], y=EU, fill='tozeroy', text=ts_continents['Europe'], name='Europe'),
            go.Scatter(x=ts_continents['date'], y=NA, fill='tonexty', text=ts_continents['North America'], name='North America'),
            go.Scatter(x=ts_continents['date'], y=AS, fill='tonexty', text=ts_continents['Asia'], name='Asia'),
            go.Scatter(x=ts_continents['date'], y=SA, fill='tonexty', text=ts_continents['South America'], name='South America'),
            go.Scatter(x=ts_continents['date'], y=AF, fill='tonexty', text=ts_continents['Africa'], name='Africa'),
            go.Scatter(x=ts_continents['date'], y=OC, fill='tonexty', text=ts_continents['Oceania'], name='Oceania'),
        ])
        fig_contients.update_layout(font_family='IBM Plex Sans',
                            title="L'évolution du nombre des infections mondiales",
                            title_x=0.5, font_size=13,
                            margin=dict(l=10, r=10, b=10, t=80),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.00,
                                xanchor="right",
                                x=1
        ))
        st.plotly_chart(fig_contients, use_container_width=True)

        space(1)

        # Buble chart par Continent
        continents_cases = df_countries.groupby('continent_iso').sum().reset_index()[['continent_iso', '4/9/21']]
        continents_cases.rename(columns={'4/9/21':'cases'}, inplace=True)
        continents_cases.sort_values('cases', ascending=False, inplace=True)
        fig = go.Figure(data=[go.Scatter(
            x=list(continents_cases['continent_iso']), y=[1] * 6,
            text=continents_cases['cases'],
            mode='markers',
            marker=dict(
                size=continents_cases['cases'],
                color=continents_cases['cases'],
                sizemode='area',
                sizeref=2000,
                showscale=True,
                colorscale='Plasma_r',
                colorbar=dict(
                    title="Infections",
                ),
            )
        )])
        fig.update_layout(font_family='IBM Plex Sans',
                        title='Les infections par continents',
                        title_x=0.5, font_size=13,
                        margin=dict(l=10, r=10, b=10, t=60),
                        plot_bgcolor='rgba(0,0,0,0)'
                        )
        fig.update_yaxes(title=None, showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

        country = 'cases'
        temp = cas_quotidiens(ts_global[['date', country]])
        fig = go.Figure([
            go.Bar(
                x=temp['date'],
                y=temp['evo'],
                name='Cas Quotidiens'
            ),
            go.Scatter(
                x=temp['date'],
                y=temp['evo7j'],
                line=dict(color='rgb(31, 119, 180)'),
                name='Moyenne sur 7 jours glissants'
            ),
        ])
        fig.update_layout(
            font_family='IBM Plex Sans',
            title="Nombre de nouveaux cas quotidiens",
            title_x=0.5, font_size=13,
            margin=dict(l=10, r=10, b=10, t=80),
            yaxis=dict(
                title="Nombre de cas"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.00,
                xanchor="right",
                x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        evo_range = [-20, 20] if limit_range else [-50, 70]
        data = evolution_couleur(ts_global[['date', country]])
        fig = go.Figure(data=[go.Bar(
            x=data['date'],
            y=data['pourcent'],
            marker_color=data['color'],
        )])
        fig.update_layout(
            font_family='IBM Plex Sans',
            title_x=0.5, font_size=13,
            margin=dict(l=10, r=10, b=10, t=80),
            title_text="Taux de croissance des infections (sur une moyenne de 7 jours)",
            yaxis=dict(
                range=evo_range, 
                title="Pourcentage d'évolution"),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif sub_section == 'Comparateur de Continents':
        st.header("Compararer les tendences et saisonalités par continents")
        with st.sidebar.expander('Options de présentation'):
            limit_range = st.checkbox('Graphiques resserrés')
        space(1)    
        df_cont = ts_continents     
        cols_to_plot = st.multiselect(
         'Choisissez un ou plusieurs continents:',
         ['Africa', 'Asia', 'Europe', 'North America', 'Océania', 'South America'],['Africa'])
        
        df_cont['date'] = pd.to_datetime(df_cont['date'])
        df_cont = df_cont.set_index('date')   
        fig = px.line(df_cont, x=df_cont.index, y= cols_to_plot, labels={
                         "date": "Date (jours)",
                         "value": "Cumul des cas",
                         "variable": "Continent"})   
        fig.update_layout(
                font_family='IBM Plex Sans',
                title="Cumul des cas quotidiens",
                title_x=0.5, font_size=13,
                margin=dict(l=10, r=10, b=10, t=80),
                yaxis=dict(
                    title="Nombre de cas"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.00,
                    xanchor="right",
                    x=1),
            )   
        st.plotly_chart(fig,use_container_width=True,sharing="streamlit")

        col1, col2 = st.columns(2)
        with col1:
            diff_1 = df_cont.diff()
            fig2 = px.line(diff_1, x=diff_1.index, y= cols_to_plot, labels={
                            "date": "Date (jours)",
                            "value": "Nouveaux cas / jours",
                            "variable": "Continent"})
            fig2.update_layout(
                font_family='IBM Plex Sans',
                title="Nouveaux cas quotidiens",
                title_x=0.5, font_size=13,
                margin=dict(l=10, r=10, b=10, t=80),
                yaxis=dict(
                    title="Nombre de cas"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.00,
                    xanchor="right",
                    x=1),
            )   
            st.plotly_chart(fig2,use_container_width=True,sharing="streamlit")

        with col2:
            weekly_mean = diff_1.resample('W').mean()
            fig3 = px.line(weekly_mean, x=weekly_mean.index, y= cols_to_plot, labels={
                            "date": "Date (jours)",
                            "value": "Nouveaux cas / semaines",
                            "variable": "Continent"})
            fig3.update_layout(
                font_family='IBM Plex Sans',
                title="Cas quotidiens (lissés sur 7 jours)",
                title_x=0.5, font_size=13,
                margin=dict(l=10, r=10, b=10, t=80),
                yaxis=dict(
                    title="Nombre de cas"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.00,
                    xanchor="right",
                    x=1),
            )   
            st.plotly_chart(fig3,use_container_width=True,sharing="streamlit")

        st.header('Saisonnalité')
        diff_1['Year'] = df_cont.index.year
        diff_1['Month'] = df_cont.index.month
        for conti in cols_to_plot:
            fig4 = px.box(diff_1, x="Month", y=conti)
            st.plotly_chart(fig4,use_container_width=True,sharing="streamlit")

    elif sub_section == 'Cartographie':
            geo_type = st.sidebar.selectbox(
                'Type de carte',
                ('world', 'usa', 'europe', 'asia', 'africa', 'north america', 'south america'),        
            )  
            series = ts_countries
            series_continent = ts_continents
            pop_mondial = pop_mondiale
            country_code = countries_code
            final_country_code = country_code.copy()
            final_country_code  = final_country_code[["ISO3 CODE","LABEL EN","LABEL FR","geo_point_2d"]]
            final_country_code.rename(columns={"ISO3 CODE":"Country Code"}, inplace=True)
            final_pop_mondial = pop_mondial.copy()
            final_pop_mondial = final_pop_mondial[['Country Name', 'Country Code', 'Indicator Name','2020']]
            final_pop_mondial["Country Name"].dropna(inplace=True)
            final_covidos= pd.melt(series, id_vars=["date"],
                                var_name="LABEL EN",
                                value_name="Covidés")
            final_covidos = final_covidos.sort_values(by=("date"))
            final_country_code = final_country_code.merge(final_pop_mondial, how="inner", on="Country Code")
            final_covid19 = final_covidos.merge(final_country_code, how="left", on="LABEL EN")
            final_covid19['time'] = final_covid19['date'].apply(lambda x: x.date()).apply(str)

            fig = px.choropleth(
                final_covid19,
                locations="LABEL EN",  
                color=np.log(final_covid19["Covidés"]), 
                hover_name="LABEL EN",  
                animation_frame="time",
                locationmode="country names",
                scope=geo_type,
                color_continuous_scale="orrd", height=700)
            fig.update(layout_coloraxis_showscale=False)
            fig.update_layout(
                dragmode=False,
                margin=dict(l=10, r=10, b=10, t=80))
            st.plotly_chart(fig,use_container_width=True,sharing="streamlit")

    elif sub_section == 'Evolution par Pays':
        st.header("Evolution fine de l'épidémie par pays")
        with st.sidebar.expander('Options de présentation'):
            limit_range = st.checkbox('Graphiques resserrés')
        space(1)

        continent = st.selectbox('Choisissez le continent à étudier : ', 
                                 df_countries['continent_iso'].dropna().unique())
        list_countries = df_countries[df_countries['continent_iso'] == continent]['Country/Region'].unique()
        col1, col2 = st.columns([1, 2])
        with col1:
            space(2)
            st.markdown('''
                Etude fine des contaminations des pays composant un continent. 
                Selectionner un continent puis le pays pour voir l'évolution de l'épidémie 
                dans ce denier.
            ''')
            space(1)
            st.metric(label="Nombre de pays", value=len(list_countries))
            space(1)
            st.metric(label="Nombre d'infections", value=df_countries[df_countries['Country/Region'].isin(list_countries)].sum()[1:-3].sum())

        with col2:
            fig = px.area(ts_countries, x='date', y=list_countries)
            fig.update_layout(font_family='IBM Plex Sans',
                              title=f"L'évolution du nombre des infections dans le continent : {continent}",
                              title_x=0.5, font_size=13,
                              margin=dict(l=10, r=10, b=10, t=80),
                              showlegend=False,
                              yaxis=dict(title='Nombre de cas')
            )
            st.plotly_chart(fig, use_container_width=True)

        space(1)
        st.header("Analyse de l'évolution de l'épidémie dans un pays")
        country = st.selectbox('Choisissez le pays à étudier : ', 
                               list_countries)

        fig = px.area(ts_countries, x='date', y=country)
        fig.update_layout(
            font_family='IBM Plex Sans',
            title=f"Sommes des cas positifs en {country}",
            title_x=0.5, font_size=13,
            margin=dict(l=10, r=10, b=10, t=80),
            showlegend=False,
            yaxis=dict(title='Nombre de cas')
        )
        st.plotly_chart(fig, use_container_width=True)

        temp = cas_quotidiens(ts_countries[['date', country]])
        fig = go.Figure([
            go.Bar(
                x=temp['date'],
                y=temp['evo'],
                name='Cas Quotidiens'
            ),
            go.Scatter(
                x=temp['date'],
                y=temp['evo7j'],
                line=dict(color='rgb(31, 119, 180)'),
                name='Moyenne sur 7 jours glissants'
            ),
        ])
        fig.update_layout(
            font_family='IBM Plex Sans',
            title="Nombre de nouveaux cas quotidiens",
            title_x=0.5, font_size=13,
            margin=dict(l=10, r=10, b=10, t=80),
            yaxis=dict(
                title="Nombre de contamnisations"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.00,
                xanchor="right",
                x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        evo_range = [-50, 50] if limit_range else [-200, 200]
        data = evolution_couleur(ts_countries[['date', country]])
        fig = go.Figure(data=[go.Bar(
            x=data['date'],
            y=data['pourcent'],
            marker_color=data['color'],
        )])
        fig.update_layout(
            font_family='IBM Plex Sans',
            title_x=0.5, font_size=13,
            margin=dict(l=10, r=10, b=10, t=80),
            title_text="Taux de croissance des infections (sur une moyenne de 7 jours)",
            yaxis=dict(
                range=evo_range, 
                title="Pourcentage d'évolution"),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif sub_section == 'Hot Zone':
        st.header("Où dans le monde les contagions ont été les plus rapides ?")
        space(1)
        
        st.subheader('Records de croissance toutes périodes')
        df_count = ts_countries
        df_count['date'] = pd.to_datetime(df_count['date'])
        df_count = df_count.set_index('date')
        df_count = df_count.diff()
        monthly_sum = df_count.resample('M').sum()
        monthly_growth = pd.DataFrame()
        for country in monthly_sum.columns:
            monthly_growth[country] = monthly_sum[country].pct_change()
        monthly_growth.replace(to_replace=np.inf,value=0, inplace = True)
        monthly_growth['Croissance (%)']= monthly_growth.max(axis=1)
        monthly_growth['Pays au plus fort taux de croissance'] = monthly_growth.idxmax(axis=1)
        monthly_growth['Year'] = monthly_growth.index.year
        monthly_growth['Month'] = monthly_growth.index.month
        st.table(monthly_growth[['Pays au plus fort taux de croissance', 
                                 'Month',
                                 'Year', 
                                 'Croissance (%)',]].nlargest(5,'Croissance (%)'))
        
        space(1)

        st.subheader('Records sur une période selectionnée')

        col1, col2 = st.columns(2)
        with col1:
            year_to_plot = st.selectbox(
                'Choisissez une année:',
                ('2020','2021'))
        with col2:
            month_to_plot = st.number_input(
            'Choisissez un mois:',
            min_value=1,
            max_value=12)
        st.table(monthly_growth[['Croissance (%)',
                                 'Pays au plus fort taux de croissance',
                                 'Month','Year']]
                 [(monthly_growth['Year'] == int(year_to_plot))&
                  (monthly_growth['Month'] == int(month_to_plot))].nlargest(5,'Croissance (%)'))

if section == 'Prediction':

    st.title("Construction d'un modèle Prédictif")

    sub_section = st.sidebar.radio(
        'Selection de la sous-section :',
        ['ARIMA vs PROPHET', 
         'Back to the future'
        ])
    st.sidebar.title(' ')

    if sub_section == 'ARIMA vs PROPHET':
        with st.sidebar.expander('Options du modèle ARIMA'):
            p = st.slider('Valeur : p', 0, 20, 1)
            d = st.slider('Valeur : d', 0, 10, 2)
            q = st.slider('valeur : q', 0, 10, 0)

        space(1)
        st.markdown('''
            Comparaison de deux modèles de machine learning qui prendrons en compte l'historique des
            contaminations passées pour **prévoir les contaminations du 10 avril 2020 au 30 avril 2020.**
        ''')
        space(1)
        df_train = ts_global[ts_global['date'] < '2020-04-10']
        df_test = ts_global[ts_global['date'].between('2020-04-10','2020-04-30')]

        mean_test = df_test['cases'].mean()

        # ARIMA
        df_arima = df_train.set_index('date')
        df_arima.index = df_arima.index.to_period('D')
        arima_model = ARIMA(df_arima['cases'], order=(p, d, q))
        arima_model = arima_model.fit()
        detail_arima = arima_model.summary()

        forecast = arima_model.get_forecast(21)
        yhat = pd.DataFrame(forecast.predicted_mean)
        yhat_conf_int = forecast.conf_int(alpha=0.05)

        arima_forecast = yhat.merge(yhat_conf_int, right_index=True, left_index=True)
        arima_forecast['true'] = df_test['cases'].values
        arima_forecast.rename(columns={'predicted_mean':'yhat',
                                    'lower cases':'yhat_lower',
                                    'upper cases':'yhat_upper' }, 
                                    inplace=True)

        arima_rmse = mean_squared_error(arima_forecast['yhat'].values, df_test['cases'], squared=False)

        # PROPHET
        df_prophet = df_train.rename(columns={'date':'ds', 'cases':'y'})
        prophet_model = Prophet()
        prophet_model.fit(df_prophet)
        future = prophet_model.make_future_dataframe(periods=21)
        forecast = prophet_model.predict(future)
        prophet_forecast = forecast[forecast['ds'].between('2020-04-10','2020-04-30')]
        prophet_forecast = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        prophet_forecast['true'] = df_test['cases'].values

        prophet_rmse = mean_squared_error(prophet_forecast['yhat'].values, df_test['cases'], squared=False)

        # PRINT
        if arima_rmse > prophet_rmse:
            arima_color, prophet_color = 'inverse', 'normal'
        else:
            arima_color, prophet_color = 'normal', 'inverse'
        col1, col2 = st.columns(2)
        with col1:
            st.header("ARIMA")
            space(1)
            st.metric('Moyenne des erreurs sur la période',
                      round(arima_rmse), 
                      f'{round((arima_rmse*100)/mean_test, 2)} %', 
                      delta_color=arima_color)
            space(1)
            st.markdown(f'''
                La moyenne des infections réelles sur la période est de {round(mean_test)} cas, 
                soit un taux d'erreur de {round((arima_rmse*100)/mean_test, 2)}%
                ''')
            with st.expander('Prévision par ARIMA'):
                st.dataframe(arima_forecast)
            
        with col2:
            st.header('PROPHET')
            space(1)
            st.metric('Moyenne des erreurs sur la période',
                      round(prophet_rmse), 
                      f'{round((prophet_rmse*100)/mean_test, 2)} %', 
                      delta_color=prophet_color)
            space(1)
            st.markdown(f'''
                La moyenne des infections réelles sur la période est de {round(mean_test)} cas, 
                soit un taux d'erreur de {round((prophet_rmse*100)/mean_test, 2)}%
                ''')
            with st.expander('Prévision par PROPHET'):
                st.dataframe(prophet_forecast)

        space(1)
        with st.expander('Détail du modèle ARIMA'):
            detail_arima
        
        with st.expander('Diagnostique du modèle ARIMA'):
            st.pyplot(arima_model.plot_diagnostics(figsize=(16,12)))
        
    if sub_section == 'Back to the future':

        st.header('Selection des paramètres')
        col1, col2 = st.columns([1, 2])
        with col1:
            pred_type = st.radio(
                "Selection du type de prévision",
                ('Globale', 'Continent', 'Pays'))
            st.session_state.start = st.button('Valider')
        with col2:
            pred_date = st.date_input(
                "Selection de la période à prévoir",
                value=date(2021, 5, 31), 
                min_value=date(2021, 4, 11))
            if pred_type == 'Continent':
                continent = st.selectbox(
                    'Selection du continent',
                    df_countries['continent_iso'].unique())
            elif pred_type == 'Pays':
                pays = st.selectbox(
                    'Selection du Pays',
                    df_countries['Country/Region'].unique())    
        st.markdown('---') 

        if st.session_state.start:
            if pred_type == 'Pays':
                df_temp = ts_countries[['date', pays]].copy()
                df_temp.rename(columns={pays:'cases'}, inplace=True)
            elif pred_type == 'Continent':
                df_temp = ts_continents[['date', continent]].copy()
                df_temp.rename(columns={continent:'cases'}, inplace=True)
            else:
                df_temp = ts_global.copy()

            df_pred = df_temp.set_index('date')
            with st.spinner('En cours de calculs'):
                stepwise_fit = auto_arima(df_pred['cases'],
                          trace=True,
                          suppress_warinings=True)
            p_val = stepwise_fit.order[0]
            d_val = stepwise_fit.order[1]
            q_val = stepwise_fit.order[2]
            st.success(f'La recherche automatique des meilleurs paramêtres donne : ({p_val}, {d_val}, {q_val})')

            # Entrainement d'ARIMA
            pred_model = ARIMA(df_pred['cases'], order=(p_val, d_val, q_val))
            pred_model = pred_model.fit()

            d0 = date(2021, 4, 10)
            nb_days = pred_date - d0
            forecast = pred_model.get_forecast(nb_days.days)
            yhat = pd.DataFrame(forecast.predicted_mean)
            yhat_conf_int = forecast.conf_int(alpha=0.05)

            prediction = yhat.merge(yhat_conf_int, right_index=True, left_index=True)
            prediction.rename(columns={'predicted_mean':'yhat',
                                        'lower cases':'yhat_lower',
                                        'upper cases':'yhat_upper' }, 
                                        inplace=True)

            prediction = prediction.apply(round)
            prediction.rename(columns={'yhat':'cases', 'yhat_lower':'cases_lower', 'yhat_upper':'cases_upper'}, inplace=True)

            pred = pd.concat([df_pred, prediction]).sort_index()
            pred['cases_lower'].fillna(pred['cases'], inplace=True)
            pred['cases_upper'].fillna(pred['cases'], inplace=True)

            fig = go.Figure([
                go.Scatter(
                    name='Measurement',
                    x=pred.index,
                    y=pred['cases'],
                    mode='lines',
                    line=dict(color='rgb(31, 119, 180)'),
                    showlegend=False
                ),
                go.Scatter(
                    name='Upper Bound',
                    x=pred.index,
                    y=pred['cases_upper'],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ),
                go.Scatter(
                    name='Lower Bound',
                    x=pred.index,
                    y=pred['cases_lower'],
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=False
                )
            ])
            fig.update_layout(
                yaxis_title='Nombre de cas',
                title=f'Prédiction des cas du 2021-04-10 au {pred_date}',
                hovermode="x"
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                space(1)
                st.write('Détail des prévisions')
                space(1)
                st.dataframe(prediction['cases'])
            with col2:
                fig = go.Figure([
                    go.Scatter(
                        name='Measurement',
                        x=prediction.index,
                        y=prediction['cases'],
                        mode='lines',
                        line=dict(color='rgb(31, 119, 180)'),
                        showlegend=False
                    ),
                    go.Scatter(
                        name='Upper Bound',
                        x=prediction.index,
                        y=prediction['cases_upper'],
                        mode='lines',
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        showlegend=False
                    ),
                    go.Scatter(
                        name='Lower Bound',
                        x=prediction.index,
                        y=prediction['cases_lower'],
                        marker=dict(color="#444"),
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(68, 68, 68, 0.3)',
                        fill='tonexty',
                        showlegend=False
                    )
                ])
                fig.update_layout(
                    yaxis_title='Nombre de cas',
                    title='Focus sur la période prédite',
                    hovermode="x"
                )
                st.plotly_chart(fig, use_container_width=True)

                # on laisse une semaine de vérification sur RMSE
                # On sort les stats RMSE + diag en extend

st.sidebar.title(' ')
st.sidebar.info('Data founie par le **Johns Hopkins Coronavirus Resource Center**')