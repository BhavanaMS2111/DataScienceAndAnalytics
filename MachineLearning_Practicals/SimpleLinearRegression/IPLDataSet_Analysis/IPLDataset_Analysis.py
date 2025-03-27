import pandas as pd
import streamlit as st
import base64

# Load datasets
player_details = pd.read_csv(r'D:\Data Science and Machine Learning\IPLDataSet_Analysis\2024_players_details.csv')
ball_by_ball = pd.read_csv(r'D:\Data Science and Machine Learning\IPLDataSet_Analysis\Ball_By_Ball_Match_Data.csv')
match_info = pd.read_csv(r'D:\Data Science and Machine Learning\IPLDataSet_Analysis\Match_Info.csv')
teams_info = pd.read_csv(r'D:\Data Science and Machine Learning\IPLDataSet_Analysis\teams_info.csv')

# Function to add background image with opacity
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        
        /* Adding a semi-transparent overlay */
        .overlay {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 10px;
            border-radius: 10px;
            width: fit-content;
            display: inline-block;
        }}

        /* Styling for header and dropdown */
        h1, h2, h3, label {{
            color: white !important;
            font-weight: bold;
            text-shadow: 2px 2px 4px black;
        }}

        /* Styling for dropdown */
        .stSelectbox div[data-baseweb="select"] {{
            background: rgba(255, 255, 255, 0.8);
            border-radius: 5px;
            padding: 5px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to add background
add_bg_from_local("ipl_background.jpg")

# Title with better readability
st.markdown('<h1 class="overlay">🏏 Historic IPL Data Analysis</h1>', unsafe_allow_html=True)

# Dropdown menu with styling
dataset_selection = st.selectbox(
    '📊 Select which Data you would like to Analyze:',
    ['Player Details', 'Ball By Ball Data', 'Match Info', 'Teams Info']
)

# Define relevant columns for each dataset with renamed headers
relevant_columns = {
    "Player Details": {
        "Name": "Player Name", 
        "battingName": "Batting Style", 
        "fieldingName": "Fielding Style", 
        "imgUrl": "Profile Image"
    },
    "Ball By Ball Data": {
        "match_id": "Match ID", 
        "over": "Over Number", 
        "ball": "Ball Number", 
        "batsman": "Batsman Name", 
        "bowler": "Bowler Name", 
        "runs_batsman": "Runs Scored"
    },
    "Match Info": {
        "match_number": "Match Number", 
        "team1": "Team 1", 
        "team2": "Team 2", 
        "match_date": "Match Date", 
        "toss_winner": "Toss Winner"
    },
    "Teams Info": {
        "team_name": "Team Name", 
        "captain": "Captain", 
        "coach": "Coach"
    }
}

# Dataset dictionary
dataset_dict = {
    "Player Details": player_details,
    "Ball By Ball Data": ball_by_ball,
    "Match Info": match_info,
    "Teams Info": teams_info
}

# Show selected dataset with a clear title
st.markdown(f'<h2 class="overlay">📌 Showing Data for: {dataset_selection}</h2>', unsafe_allow_html=True)

# Extract relevant columns and rename them
df_selected = dataset_dict[dataset_selection].rename(columns=relevant_columns[dataset_selection])

# Display dataframe with renamed columns
st.dataframe(df_selected)
