import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hydralit_components as hc
import plotly.graph_objects as go
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
#import imblearn
#from imblearn.combine import SMOTETomek
#from imblearn.under_sampling import TomekLinks
import requests
from streamlit_lottie import st_lottie
import json

#defining lottie function to visualize animated pictures
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

df = pd.read_csv("CleanCustomerChurn.csv")

def upload():
    uploaded_file = st.file_uploader("Upload the Updated Dataset")
    if uploaded_file is None: 
        df = pd.read_csv("CleanCustomerChurn.csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

#setting configuration of the page and expanding it
st.set_page_config(layout='wide', initial_sidebar_state='collapsed', page_title='Customer Churn Analysis')
st.expander('Expander')


#creating menu data which will be used in navigation bar specifying the pages of the dashboard
menu_data = [
    {'label': "Home", 'icon': 'bi bi-house'},
    {'label': 'Data', 'icon': 'bi bi-table'},
    {'label':"Overview", 'icon':'bi bi-people'},
    {'label':"Customer Behavior", 'icon': 'bi bi-person-workspace'},
    {'label':"Churn Analysis", 'icon': 'bi bi-person-x'},
    {'label': 'Profile', 'icon': 'bi bi-person-lines-fill'},
    {'label':"Application", 'icon': 'bi bi-person-bounding-box'},
    ]

over_theme = {'txc_inactive': 'white','menu_background':'rgb(180,151,231)', 'option_active':'white'}


#inserting hydralit component: nagivation bar
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=True,
    sticky_nav=True, #at the top or not
    sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
)




#editing first page of the dashboard with images, titles, and text
if menu_id == 'Home':
    col1, col2 = st.columns(2)
    #col1.image('churn.png')
    with col1:
        lottie_home= load_lottiefile("customer.json") #('https://assets9.lottiefiles.com/packages/lf20_xt3zjwah.json')
        st_lottie(lottie_home)

    with col2:
        st.title('Customer Churn Analysis')
        st.write('Customer Churn, or attrition rate, is a rate that indicates the loss of a customer. It is when a customer decides to stop doing business with an entity. Churn rate is an important factor to evaluate in order to analyse the performance of an entity.')
        m = st.markdown("""
        <style>
            div.stButton > button:first-child {
            color: #fff;
            background-color: rgb(180,151,231);
            }
        </style>""", unsafe_allow_html=True)
        b = st.button("Start exploring!")



#BREAK



#editing second page which is about the data
if menu_id == 'Data':
    col1, col2 = st.columns([2,1])

    with col2:
        lottie_data= load_lottiefile("data.json")
        st_lottie(lottie_data)

    with col1:
        st.title("Data View")
        st.markdown("""
        <style>
        .change-font {
        font-size:15px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="change-font">Every user is assigned a prediction value that estimates their state of churn at any given time. This value is based on: User demographic information, Browsing behavior, and Historical purchase data among other information. </p>', unsafe_allow_html=True)
        st.markdown('<p class="change-font">It factors in our unique and proprietary predictions of how long a user will remain a customer. This score is updated every day for all users who have a minimum of one conversion. The values assigned are between 1 and 5. </p>', unsafe_allow_html=True)


    #add option to choose own data or given data defined earlier
    upload()
    st.write('Or you can use the data already built-in!')





#BREAK



if menu_id == 'Overview':


    #creating new dataframes to control the interactive text
    df1= df.groupby(['churn_risk_score', 'gender']).size().reset_index(name='counts')
    df1 = pd.pivot_table(df1, values=['counts'], columns=None, index=['gender'], aggfunc='sum', sort=True)
    df1 = df1.reset_index()
    
    #small editing to the age column
    age2 = {'Oct-19': '10 to 19', '20-29':'20 to 29', '30-39':'30 to 39', '40-49':'40 to 49', '50-59':'50 to 59', '60-69':'60 to 69'}
    df=df.replace({'age2': age2})
    
    df2= df.groupby(['churn_risk_score', 'age2']).size().reset_index(name='counts')
    df2 = pd.pivot_table(df2, values=['counts'], columns=None, index=['age2'], aggfunc='sum')
    df2 = df2.reset_index()
    sorted2 = df2.sort_values(by='counts', ascending=False).reset_index()

    df3= df.groupby(['churn_risk_score', 'membership_category']).size().reset_index(name='counts')
    df3 = pd.pivot_table(df3, values=['counts'], columns=None, index=['membership_category'], aggfunc='sum')
    df3 = df3.reset_index()
    sorted3 = df3.sort_values(by='counts', ascending=False).reset_index()

    df4= df.groupby(['medium_of_operation', 'churn_risk_score']).size().reset_index(name='counts')
    df4 = pd.pivot_table(df4, values=['counts'], columns=None, index=['medium_of_operation'], aggfunc='sum')
    df4 = df4.reset_index()
    sorted4 = df4.sort_values(by='counts', ascending=False).reset_index()

    df5 = df.groupby(['churn_risk_score', 'preferred_offer_types']).size().reset_index(name='counts')
    df5 = pd.pivot_table(df5, values=['counts'], columns=None, index=['preferred_offer_types'], aggfunc='sum', sort=True)
    df5 = df5.reset_index()
    sorted5 = df5.sort_values(by='counts', ascending=False).reset_index()

    df6 = df.groupby(['churn_risk_score', 'region_category']).size().reset_index(name='counts')
    df6 = pd.pivot_table(df6, values=['counts'], columns=None, index=['region_category'], aggfunc='sum', sort=True)
    df6 = df6.reset_index()
    sorted6 = df6.sort_values(by='counts', ascending=False).reset_index()


    #splitting page into 4 columns for animations
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        #animation1 about gender dispersion
        lottie_gender= load_lottiefile("gender.json")
        st_lottie(lottie_gender, height=150, width=200)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                 The female customer base is at {df1['counts'].loc[df1['gender'] == 'F'].values[0]} customers, while the Male customer base is at {df1['counts'].loc[df1['gender'] == 'M'].values[0]} customers
                </div>""",unsafe_allow_html = True)


    with col2:
        #animation2 about membership category
        lottie_member= load_lottiefile("member.json")
        st_lottie(lottie_member, height=150, width=200)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                Most of the customers have {sorted3['membership_category'].loc[0]}
                </div>""",unsafe_allow_html = True)


    with col3:
        #animation3 about offer preferred
        lottie_offer= load_lottiefile("offer.json")
        st_lottie(lottie_offer, height=150, width=200)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                Most of the customers prefer {sorted5['preferred_offer_types'].loc[0]} as offer types
                </div>""",unsafe_allow_html = True)

    with col4:
        #animation4 about region dispersion of customers
        lottie_region= load_lottiefile("location.json")
        st_lottie(lottie_region, height=150, width=200)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                Most of the customers come from the {sorted6['region_category'].loc[0]}
                </div>""",unsafe_allow_html = True)


    #empty space for clearer view
    st.empty()
    st.write('')
    st.write('')
    st.write('')


    #line dividing animations with plots
    theme_override = {'bgcolor': 'rgb(220,176,242)','title_color': 'white','content_color': 'white','progress_color': ' rgb(220,176,242)',
    'icon_color': 'white', 'icon': 'bi bi-calendar', 'content_text_size' : '10%'}
    hc.progress_bar(content_text= 'Demographics', override_theme=theme_override)

    #splitting again for plots
    col1, col2= st.columns(2)

    with col1:
        #plot1 about gender distribution in customers
        df1= df.groupby(['churn_risk_score', 'gender']).size().reset_index(name='counts')
        df1 = pd.pivot_table(df1, values=['counts'], columns=None, index=['gender'], aggfunc='sum', sort=True)
        df1 = df1.reset_index()
        fig = go.Figure(px.pie(df1, values='counts', names='gender',
                 color_discrete_sequence= ['rgb(158,185,243)', 'rgb(254,136,177)'],
                 labels= {'gender': 'Gender',
                         'counts':'Number of Customers'}
                ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=500,
                            title='Gender Distribution',
                            title_font_size=24)
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)



    with col2:
    #plot2
        df2= df.groupby(['churn_risk_score', 'age2']).size().reset_index(name='counts')
        df2 = pd.pivot_table(df2, values=['counts'], columns=None, index=['age2'], aggfunc='sum', sort=True)
        df2 = df2.reset_index()
        fig = go.Figure(px.treemap(df2, path=['age2'], values='counts',
            color_discrete_sequence= ['rgb(220,176,242)', ' rgb(180,151,231)', 'rgb(158,185,243)', ' rgb(254,136,177)',
            'rgb(136,204,238)', 'rgb(247,129,191)'],
         labels= {'churn_risk_score': 'Churn Score',
                 'counts':'Number of Customers'}
                 ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=600,
            title='Age Groups',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                 The highest age group is the ages {sorted2['age2'].loc[0]}
                </div>""",unsafe_allow_html = True)


    st.empty()
    st.write('')
    st.write('')
    st.write('')

    #divide to seperate sections
    theme_override = {'bgcolor': 'rgb(220,176,242)','title_color': 'white','content_color': 'white','progress_color': ' rgb(220,176,242)',
    }
    hc.progress_bar(content_text= 'Options Available', override_theme=theme_override)

    #split to plot 2 side-by-side graphs
    col3, col4 = st.columns(2)

    with col3:
        #plot3
        df3 = df.groupby(['churn_risk_score', 'membership_category']).size().reset_index(name='counts')
        df3['churn_risk_score'] = df3['churn_risk_score'].astype(str)
        fig = go.Figure(px.histogram(df3, x='membership_category', y='counts',barmode='group',
            color_discrete_sequence= ['rgb(158,185,243)'],
             labels= {'counts':'Number of Customers',
                     'membership_category': 'Membership'}))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=600,
            title='Membership Categories Available',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)



    with col4:
    #plot4
        df4= df.groupby(['medium_of_operation', 'age2']).size().reset_index(name='counts')
        fig = go.Figure(px.histogram(df4, x='medium_of_operation', y='counts', barmode='group',
            color_discrete_sequence= ['rgb(247,129,191)'],
             labels= {'medium_of_operation': 'Medium of Operation',
                     'counts':'Number of Customers'}))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=600,
            title='Mediums of Operation',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                Most of the customers use {sorted4['medium_of_operation'].loc[0]} as medium of operation
                </div>""",unsafe_allow_html = True)



    st.empty()
    st.write('')
    st.write('')
    st.write('')

    #seperate sections
    theme_override = {'bgcolor': 'rgb(220,176,242)','title_color': 'white','content_color': 'white','progress_color': ' rgb(220,176,242)',
    }
    hc.progress_bar(content_text= 'Dispersion', override_theme=theme_override)

    #split fot side-by-side plots
    col5, col6 = st.columns(2)

    with col5:
    #plot5
        df5 = df.groupby(['churn_risk_score', 'preferred_offer_types']).size().reset_index(name='counts')
        df5 = pd.pivot_table(df5, values=['counts'], columns=None, index=['preferred_offer_types'], aggfunc='sum', sort=True)
        df5 = df5.reset_index()
        fig = go.Figure(px.pie(df5, values='counts', names='preferred_offer_types',
             color_discrete_sequence= ['rgb(220,176,242)', ' rgb(180,151,231)', 'rgb(158,185,243)', ' rgb(254,136,177)',
             'rgb(136,204,238)', 'rgb(247,129,191)'],
             labels= {'preferred_offer_types': 'Offer Types Preferred',
                     'counts':'Number of Customers'}
            ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=600,
            title='Offer Types Preferred',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)


    with col6:
        #plot6
        df6 = df.groupby(['churn_risk_score', 'region_category']).size().reset_index(name='counts')
        df6 = pd.pivot_table(df6, values=['counts'], columns=None, index=['region_category'], aggfunc='sum', sort=True)
        df6 = df6.reset_index()
        fig = go.Figure(px.pie(df6, values='counts', names='region_category',
             color_discrete_sequence= ['rgb(220,176,242)', ' rgb(254,136,177)',
             'rgb(136,204,238)'],
             labels= {'region_category': 'Region',
                     'counts':'Number of Customers'}
            ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=500,
            title='Region Dispersion',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)






if menu_id == 'Customer Behavior':

    #subsets of data to control interactivity and plots
    df7= df.groupby(['churn_risk_score', 'complaint_status']).size().reset_index(name='counts')
    df7 = pd.pivot_table(df7, values=['counts'], columns=None, index=['complaint_status'], aggfunc='sum', sort=True)
    df7 = df7.reset_index()

    df8= df.groupby(['churn_risk_score', 'past_complaint']).size().reset_index(name='counts')
    df8 = pd.pivot_table(df8, values=['counts'], columns=None, index=['past_complaint'], aggfunc='sum', sort=True)
    df8 = df8.reset_index()

    df11 = df.groupby(['churn_risk_score']).size().reset_index(name='counts')
    sorted11 = df11.sort_values(by='counts', ascending=False).reset_index()


    #split for animations
    col1, col2, col3, col4= st.columns(4)

    with col1:
        #animation5
        lottie_complaint= load_lottiefile("feedback2.json")
        st_lottie(lottie_complaint, height=150, width=200)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                Approximately {round((df7['counts'].loc[df7['complaint_status'] == 'Unsolved'] / df7['counts'].sum() *100).values[0], 1)}% of the complaints were Unsolved
            </div>""",unsafe_allow_html = True)


    with col2:
        #animation6
        lottie_wallet= load_lottiefile("wallet.json")
        st_lottie(lottie_wallet, height=150, width=200)
        st.caption(f"""
        <div>
            <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
            Customers have on average {round(df['points_in_wallet'].mean())} points in their application wallets
        </div>""",unsafe_allow_html = True)



    with col3:
        #animation7
        lottie_time_spent= load_lottiefile("time_spent.json")
        st_lottie(lottie_time_spent, height=150, width=200)
        st.caption(f"""
        <div>
            <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
            Customers spend on average {round(df['avg_time_spent'].mean())} hours on the application over the months
        </div>""",unsafe_allow_html = True)


    with col4:
        #animation8
        lottie_risk= load_lottiefile("risk.json")
        st_lottie(lottie_risk, height=150, width=200)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                The majority of customers have a churn risk score of {sorted11['churn_risk_score'].loc[0]}
            </div>""",unsafe_allow_html = True)




    st.empty()
    st.write('')
    st.write('')
    st.write('')

    #split sections
    theme_override = {'bgcolor': 'rgb(220,176,242)','title_color': 'white','content_color': 'white','progress_color': 'rgb(220,176,242)'}
    hc.progress_bar(content_text= 'Complaints', override_theme=theme_override)

    #for side-by-side plots
    col1, col2 = st.columns(2)

    with col1:
    #plot7
        df7= df.groupby(['churn_risk_score', 'complaint_status']).size().reset_index(name='counts')
        df7 = pd.pivot_table(df7, values=['counts'], columns=None, index=['complaint_status'], aggfunc='sum', sort=True)
        df7 = df7.reset_index()
        fig = px.pie(df7, values='counts', names='complaint_status',
             color_discrete_sequence= ['rgb(220,176,242)', ' rgb(180,151,231)', 'rgb(158,185,243)', ' rgb(254,136,177)',
             'rgb(136,204,238)'],
             labels= {'churn_risk_score': 'Churn Score',
                     'counts':'Number of Customers',
                     'complaint_status': 'Complaint Status'}
            )
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), autosize=False, width=550,
            title='Complaints Status',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)'})
        st.plotly_chart(fig)


    with col2:
        #plot8
        df8= df.groupby(['churn_risk_score', 'past_complaint']).size().reset_index(name='counts')
        df8 = pd.pivot_table(df8, values=['counts'], columns=None, index=['past_complaint'], aggfunc='sum', sort=True)
        df8 = df8.reset_index()
        fig = go.Figure(px.pie(df8, values='counts', names='past_complaint',
             color_discrete_sequence= ['rgb(158,185,243)', 'rgb(254,136,177)'],
             labels= {'past_complaint': 'Past Complaints',
                     'counts':'Number of Customers'}
            ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),  width=550,
            title='Past Complaints',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                Approximately {round((df8['counts'].loc[df8['past_complaint'] == 'Yes'] / df8['counts'].sum() *100).values[0], 1)}% of the customers complained in the past
            </div>""",unsafe_allow_html = True)



    st.empty()
    st.write('')
    st.write('')
    st.write('')

    #seperate sections
    theme_override = {'bgcolor': 'rgb(220,176,242)','title_color': 'white','content_color': 'white','progress_color': 'rgb(220,176,242)'}
    hc.progress_bar(content_text= 'Monetary Value', override_theme=theme_override)

    #for side-by-side plots
    col3, col4 = st.columns(2)

    with col3:
        #plot9
            df9 = df.copy()
            df9['churn_risk_score'] = df9['churn_risk_score'].astype(str)
            fig = go.Figure(px.histogram(df9, x='points_in_wallet',
               color_discrete_sequence= ['rgb(220,176,242)'],
               labels= {'points_in_wallet': 'Points in Wallet Distribution'}
            ))
            fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=600,
                title='Points in Wallet Distribution',
                title_font_size=24)
            fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
            st.plotly_chart(fig)



    with col4:
        #plot10
        df10 = df.copy()
        df10['churn_risk_score'] = df10['churn_risk_score'].astype(str)
        fig = go.Figure(px.histogram(df10, x='avg_time_spent',
            color_discrete_sequence= ['rgb(158,185,243)'],
             labels= {'avg_time_spent': 'Average Time Spent'}
            ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=600,
            title='Time Spent on App Distribution',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)



    st.empty()
    st.write('')
    st.write('')
    st.write('')

    #seperate sections
    theme_override = {'bgcolor': 'rgb(220,176,242)','title_color': 'white','content_color': 'white','progress_color': 'rgb(220,176,242)'}
    hc.progress_bar(content_text= 'Churn Dispersion', override_theme=theme_override)

    #plot11
    df11 = df.groupby(['churn_risk_score']).size().reset_index(name='counts')
    df11['churn_risk_score'] = df11['churn_risk_score'].astype(str)
    fig = go.Figure(px.bar(df11, x='churn_risk_score', y='counts', color='churn_risk_score',
        color_discrete_sequence= ['rgb(220,176,242)', ' rgb(180,151,231)', 'rgb(158,185,243)', ' rgb(254,136,177)',
        'rgb(136,204,238)'],
         labels= {'churn_risk_score': 'Churn Score',
                 'counts':'Number of Customers'}))
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=1100,
        title='Churn Score Distribution',
        title_font_size=24)
    fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    st.plotly_chart(fig)



#BREAK



if menu_id == 'Churn Analysis':

        #divide columns for animations
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        #animation9
        lottie_membership= load_lottiefile("membership.json")
        st_lottie(lottie_membership, height=150, width=200)
        st.caption(f"""
        <div>
          <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
          Customers with Basic or No Membership are more likely to churn than those with Gold, Premium, Silver, or Platinum
        </div>""",unsafe_allow_html = True)


    with col2:
        #animation10
        lottie_pts= load_lottiefile("points.json")
        st_lottie(lottie_pts, height=150, width=200)
        st.caption(f"""
        <div>
            <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
            Riskier customers purchase less than safe customers, with the same amount of points in wallet
        </div>""",unsafe_allow_html = True)

    with col3:
        #animation11
        lottie_feedback= load_lottiefile("feedback.json")
        st_lottie(lottie_feedback, height=150, width=200)
        st.caption(f"""
        <div>
            <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
            Customers with positive feedback are less likely to churn than those with negative feedback
        </div>""",unsafe_allow_html = True)


    with col4:
        #animation12
        lottie_time= load_lottiefile("time.json")
        st_lottie(lottie_time, height=150, width=200)
        st.caption(f"""
        <div>
            <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
            Riskier customers purchase less than safe customers, with the same amount of time spent on the application
        </div>""",unsafe_allow_html = True)


    #for spacing between plots
    st.empty()
    st.write('')
    st.write('')
    st.write('')

    #seperate sections
    theme_override = {'bgcolor': ' rgb(180,151,231)','title_color': 'white','content_color': 'white','progress_color': ' rgb(180,151,231)'}
    hc.progress_bar(content_text= 'Analysis', override_theme=theme_override)

    #for side-by-side plots
    col1,col2 = st.columns(2)

    with col1:
        #plot12
        df12 = df.groupby(['churn_risk_score', 'membership_category']).size().reset_index(name='counts')
        df12['churn_risk_score'] = df12['churn_risk_score'].astype(str)
        fig = go.Figure(px.histogram(df12, x='membership_category', y='counts', color='churn_risk_score', barmode='group',
            color_discrete_sequence= ['rgb(220,176,242)', ' rgb(180,151,231)', 'rgb(158,185,243)', ' rgb(254,136,177)',
            'rgb(136,204,238)'],
             labels= {'churn_risk_score': 'Churn Score',
                     'counts':'Number of Customers',
                     'membership_category': 'Membership'}
            ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=550,
            title='Churn Score Distribution by Membership',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)

        st.write('')
        st.write('')
        st.write('')


        #plot13
        df13 = df.groupby(['churn_risk_score', 'feedback']).size().reset_index(name='counts')
        df13['churn_risk_score'] = df13['churn_risk_score'].astype(str)
        fig = go.Figure(px.histogram(df13, x='counts', y='feedback', color='churn_risk_score', barmode='group',
            color_discrete_sequence= ['rgb(220,176,242)', ' rgb(180,151,231)', 'rgb(158,185,243)', ' rgb(254,136,177)',
            'rgb(136,204,238)'],
            labels= {'churn_risk_score': 'Churn Score',
                 'counts':'Number of Customers',
                 'feedback': 'Feedback Type'}
        ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=550,
            title='Churn Score Distribution by Feedback',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)


    with col2:

        #plot14
        df14 = df.copy()
        df14['churn_risk_score'] = df14['churn_risk_score'].astype(str)
        fig = go.Figure(px.scatter(df14, x="avg_transaction_value", y="points_in_wallet", color="churn_risk_score",
                    color_discrete_sequence= ['rgb(93,105,177)', 'rgb(231,41,138)', 'rgb(254,136,177)', 'rgb(136,204,238)',
                    'rgb(180,151,231)'],
                     labels= {'churn_risk_score': 'Churn Score',
                         'counts':'Number of Customers',
                         'points_in_wallet': 'Points in Wallet',
                             'avg_transaction_value': 'Average Transaction Value'}
                ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=550,
            title='Points in Wallet vs Average Transaction Value',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)

        #for spacing between plots
        st.write('')
        st.write('')
        st.write('')

        #plot15
        df15 = df.copy()
        df15['churn_risk_score'] = df15['churn_risk_score'].astype(str)
        fig = go.Figure(px.scatter(df15, x='avg_transaction_value', y='avg_time_spent', color='churn_risk_score',
            color_discrete_sequence= ['rgb(93,105,177)', 'rgb(231,41,138)', 'rgb(254,136,177)', 'rgb(136,204,238)',
            'rgb(180,151,231)'],
             labels= {'churn_risk_score': 'Churn Score',
                 'counts':'Number of Customers',
                 'avg_time_spent': 'Average Time Spent',
                 'avg_transaction_value': 'Average Transaction Value'}
        ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=550,
            title='Average Time Spent vs Average Transaction Value',
            title_font_size=24)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)


#BREAK



if menu_id == 'Profile':

    #seperation between graphs
    col1,col2, col3 = st.columns([1,2,1])

    with col2:
        name = pd.Series(df['Name'])
        selection = st.selectbox('Select a Customer', name)
        df20 = df.loc[df['Name'] == selection]

        for i in df20['gender']:
            if i == 'F':
                #if the customer is female, return a female animated image
                lottie_girl= load_lottiefile("female.json")
                st_lottie(lottie_girl, height=360, width=550)

            else:
                #if the customer is male, return a male animated image
                lottie_guy= load_lottiefile("male.json")
                st_lottie(lottie_guy, height=350, width=550)


        churn = df20['churn_risk_score'].values[0]

        for i in df20['churn_risk_score']:
            if i == 1:
                st.caption(f"""
                <div>
                    <div style="vertical-align:center;font-size:16px;padding-left:260px;padding-top:0px;margin-left:0em";>
                    Score: {i}
                </div>""",unsafe_allow_html = True)
                #if low churn risk, give highest progress then start decreasing the progress
                st.markdown("""
                    <style>
                        .stProgress .st-di {
                        background-color: rgb(139,224,164);
                        }
                    </style>
                    """, unsafe_allow_html=True)
                progress = st.progress(100)

            elif i == 2:
                st.caption(f"""
                <div>
                    <div style="vertical-align:center;font-size:16px;padding-left:260px;padding-top:0px;margin-left:0em";>
                    Score: {i}
                </div>""",unsafe_allow_html = True)
                st.markdown("""
                    <style>
                        .stProgress .st-di {
                        background-color: rgb(139,224,164);
                        }
                    </style>
                    """, unsafe_allow_html=True)
                progress = st.progress(80)

            elif i == 3:
                st.caption(f"""
                <div>
                    <div style="vertical-align:center;font-size:16px;padding-left:260px;padding-top:0px;margin-left:0em";>
                    Score: {i}
                </div>""",unsafe_allow_html = True)
                st.markdown("""
                    <style>
                        .stProgress .st-di {
                        background-color: rgb(139,224,164);
                        }
                    </style>
                    """, unsafe_allow_html=True)
                progress = st.progress(60)
            elif i == 4:
                st.caption(f"""
                <div>
                    <div style="vertical-align:center;font-size:16px;padding-left:260px;padding-top:0px;margin-left:0em";>
                    Score: {i}
                </div>""",unsafe_allow_html = True)
                st.markdown("""
                    <style>
                        .stProgress .st-di {
                        background-color: rgb(139,224,164);
                        }
                    </style>
                    """, unsafe_allow_html=True)
                progress = st.progress(40)
            else:
                st.caption(f"""
                <div>
                    <div style="vertical-align:center;font-size:16px;padding-left:260px;padding-top:0px;margin-left:0em";>
                    Score: {i}
                </div>""",unsafe_allow_html = True)
                st.markdown("""
                    <style>
                        .stProgress .st-di {
                        background-color: rgb(139,224,164);
                        }
                    </style>
                    """, unsafe_allow_html=True)
                progress = st.progress(20)


        feedback = df20['feedback'].values[0]
        theme_override = {'bgcolor': 'rgb(136,204,238)','title_color': 'white','content_color': 'white','progress_color': 'rgb(136,204,238)',
        }
        #return a bar that gives the feedback the customer entered
        hc.progress_bar(content_text= f'''Feedback: {df20['feedback'].values[0]}''', override_theme=theme_override)


    with col1:

        #info card of the customer's age
        age = df20['age'].values[0]
        theme_override = {'bgcolor': 'rgb(247,129,191)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-calendar'}
        hc.info_card(title='Age', content=str(age), theme_override=theme_override)


        #info card of the customer's region
        region = df20['region_category'].values[0]
        theme_override = {'bgcolor': 'rgb(136,204,238)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-geo-alt'}
        hc.info_card(title='Region', content=str(region), theme_override=theme_override)

        #info card of the customer's average time spent on the application
        time = df20['avg_time_spent'].values[0]
        theme_override = {'bgcolor': 'rgb(220,176,242)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-clock-fill'}
        hc.info_card(title='Time Spent', content=str(time), theme_override=theme_override)
        #st.markdown(f'Average Time Spent: {time}')



    with col3:

        #info card of the customer's points in wallet
        points = df20['points_in_wallet'].values[0]
        theme_override = {'bgcolor': 'rgb(247,129,191)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-wallet2'}
        hc.info_card(title='Wallet Points', content=str(points), theme_override=theme_override)

        #info card of the customer's membership category
        membership = df20['membership_category'].values[0]
        theme_override = {'bgcolor': 'rgb(136,204,238)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-credit-card-2-front-fill'}
        hc.info_card(title='Membership', content=str(membership), theme_override=theme_override)

        #info card of the customer's average frequency login days
        frequency = df20['avg_frequency_login_days'].values[0]
        theme_override = {'bgcolor': 'rgb(220,176,242)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-calendar-day-fill'}
        hc.info_card(title='Frequency Login', content=str(frequency) + ' Days', theme_override=theme_override)



#BREAK



if menu_id == 'Application':

    df = pd.read_csv('CleanCustomerChurn.csv')

    #seperate title and animation
    col1, col2= st.columns([2,2])
    with col1:
        st.title("Your Turn to Predict!")
        st.write("Fill out your customers' information and get their Churn Risk Score")
        st.caption(f"""
        <div>
            <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
            Examine whether your customer would have a high or low churn risk score through his behavior, demographics, and feedback. These are all key factors in predicting whether the customer would leave. To build efficient marketing campaigns and decrease uneccessary costs, it is crucial to examine this score for future customers.
        </div>""",unsafe_allow_html = True)


    with col2:
        #animation13
        lottie_app= load_lottiefile("application.json")
        st_lottie(lottie_app, height=360, width=550)


    st.empty()

    #seperate sections
    theme_override = {'bgcolor': ' rgb(180,151,231)','title_color': 'white','content_color': 'white','progress_color': ' rgb(180,151,231)'}
    hc.progress_bar(content_text= 'Fill the Form', override_theme=theme_override)

    #specifying numerical and categorical features
    numerical_features = ['age', 'days_since_last_login', 'avg_time_spent',
       'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet',
       'Tenure (months)']

    df_numerical = df[numerical_features]

    categorical_features =['gender', 'region_category', 'membership_category',
       'joined_through_referral', 'preferred_offer_types',
       'medium_of_operation', 'internet_option', 'used_special_discount',
       'offer_application_preference', 'past_complaint', 'complaint_status',
       'feedback']
    df_categorical = df[categorical_features]

       # Features
    X = pd.concat([df_categorical, df_numerical], axis = 1)

       # Target Variable
    y = df['churn_risk_score']

    #defining function to create inputs that will give a prediction
    def user_input_features():

        col3, col4= st.columns(2)
        with col3:
           Age = st.number_input("Insert the Age of the customer", value=10, min_value=10)
           Last_login = st.number_input('Insert the Number of days since the last time the customer logged in', value=1, min_value=1)
           Time_spent = st.number_input("Insert the Average time spent by the customer", value=1, min_value=1)
           Transaction = st.number_input("Insert the Average value of all transactions made by the customer", value=1, min_value=1)
           Frequency = st.number_input("Insert the Average times the customer has logged in the website", value=1, min_value=1)
           Points = st.number_input("Insert the customer's points in wallet", value=0, min_value=-1000)
           Tenure = st.number_input("Insert the tenure of the customer", value=0, min_value=0)
           Gender= st.selectbox('Select the gender of the customer', ('F', 'M'))
           Region= st.selectbox('Select the region of the customer', ('Town', 'City', 'Village'))
           Membership= st.selectbox('Select the membership category of the customer', ('Basic Membership', 'No Membership',
           'Gold Membership', 'Silver Membership', 'Premium Membership', 'Platinum Membership'))

        with col4:
           Referral = st.selectbox('Did the customer join through referral?', ('Yes', 'No', 'Maybe'))
           Offer= st.selectbox("Select the customer's preffered order type", ('Gift Vouchers/Coupons',
           'Credit/Debit Card Offers', 'Without Offers'))
           Medium = st.selectbox("Select the customer's medium of operation", ('Desktop', 'Smartphone', 'Not Specified', 'Both'))
           Internet = st.selectbox("Select the customer's internet option", ('Wi-Fi', 'Mobile_Data', 'Fiber_Optic'))
           Discount = st.selectbox('Did the customer use a special discount?', ('Yes', 'No'))
           Prefer_offer= st.selectbox('Does the customer prefer offers?', ('Yes', 'No'))
           Complaint= st.selectbox('Does the customer have any past complaints?', ('Yes', 'No'))
           Comp_stat= st.selectbox('Select the status of the complaint', ('Not Applicable', 'Unsolved', 'Solved',
           'Solved in Follow-up', 'No Information Available'))
           Feedback= st.selectbox('Select the type of feedback from the customer', ('Poor Product Quality',
           'No reason specified', 'Too many ads', 'Poor Website', 'Poor Customer Service', 'Reasonable Price',
           'User Friendly Website', 'Products always in Stock', 'Quality Customer Care'))


           data = {'age': Age, 'days_since_last_login': Last_login, 'avg_time_spent': Time_spent,
           'avg_transaction_value': Transaction, 'avg_frequency_login_days': Frequency, 'points_in_wallet': Points,
           'Tenure (months)': Tenure, 'gender': Gender, 'region_category':Region,
            'membership_category': Membership, 'joined_through_referral': Referral,
            'preferred_offer_types': Offer, 'medium_of_operation': Medium, 'internet_option': Internet,
            'used_special_discount': Discount, 'offer_application_preference': Prefer_offer,
            'past_complaint': Complaint, 'complaint_status': Comp_stat, 'feedback': Feedback}

           features= pd.DataFrame(data, index=[0])
           return features

    df1= user_input_features()

    #encoding categorical variable
    encoderlabel = LabelEncoder()
    y = encoderlabel.fit_transform(y)

    #pipeline for all necessary transformations
    cat_pipeline= Pipeline(steps=[
            ('impute', SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 'None')),
            ('ohe', OneHotEncoder(handle_unknown = 'ignore'))
            ])

    num_pipeline = Pipeline(steps=[

            ('impute', SimpleImputer(missing_values = np.nan, strategy='mean')),
            ('outlier',RobustScaler())
            ])

    column_transformer= ColumnTransformer(transformers=[
            ('ohe', cat_pipeline, categorical_features),
            ('impute', num_pipeline, numerical_features)
            ], remainder='drop')

    #chose best model based on previous trials
    model = RandomForestClassifier(class_weight='balanced')

    pipeline_model = Pipeline(steps = [('transformer', column_transformer),
                             ('model', model)])

    #train the model
    pipeline_model.fit(X, y)

    #predicting the data
    prediction = pipeline_model.predict(df1)

    m = st.markdown("""
        <style>
            div.stButton > button:first-child {
            color: #fff;
            background-color: rgb(180,151,231);
            }
        </style>""", unsafe_allow_html=True)
    submit = st.button('Predict')

    if submit:
        st.subheader(f'The Churn Risk Score of the customer is {str(prediction)}')
        st.write('---')
