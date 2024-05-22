import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
# insert image
st.sidebar.image("safe.png", width=100)

# Create a sidebar with buttons for navigation
selection = st.sidebar.radio("Go to", ("Analysis", "Prediction"))

# Based on the selected option, display different content
if selection == "Analysis":
    # Create a DataFrame by importing data from csv file
    df = pd.read_csv('dft-road-statistics-collision-last-5-years_new/dft-road-statistics-collision-last-5-years_new.csv')
    left_col1, center_col2, right_col3 = st.columns(3)
    with left_col1:
        # insert image
        st.image("car.jpg", width=150)
    with right_col3:
        # insert image
        st.image("car.jpg", width=150)

    # Title
    st.markdown(f"<h1 style='text-align: left; font-size: 20px; color: #800000;'>Analysis of Road Traffic Accidents "
                f"in UK for the year 2018-2022</h1>", unsafe_allow_html=True)
    # st.write(df)

    # columns to display input fields
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h2 style='text-align: left; font-size: 20px; color: #008080;'>Accidents by Road Type</h2>",
                    unsafe_allow_html=True)

        # Calculate value counts of road types
        road_type_counts = df['road_type'].value_counts()

        # Plotting a road type count plot
        fig, ax = plt.subplots()
        sns.countplot(y='road_type', data=df, ax=ax, palette='Paired')
        # Remove the outline
        sns.despine()
        plt.xlabel("Count")
        plt.ylabel("Road Type")

        # Display the plot using Streamlit
        st.pyplot(fig)

    with col2:
        # Create DataFrame
        data = {'No_Of_Casualities': df['number_of_casualties'],
                'accident_year': df['accident_year'],
                'urban_or_rural_area': df['urban_or_rural_area']}
        accidents_per_year = pd.DataFrame(data)

        # Filter out any rows with missing values in 'urban_or_rural_area' column
        accidents_per_year = accidents_per_year.dropna(subset=['urban_or_rural_area'])

        # Group by year and urban_or_rural_area and sum casualties
        cas_count = accidents_per_year.groupby(by=['accident_year', 'urban_or_rural_area']).sum().sort_values(
            by='No_Of_Casualities', ascending=False)

        # Plot the count of casualties per year and urban/rural area as a bar graph
        st.markdown(f"<h2 style='text-align: left; font-size: 20px; color: #008080;'>"
                    f"Number of Casualties by Year & Area</h2>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots()

        # Plot bar graph
        sns.barplot(x=cas_count.index.get_level_values('accident_year'), y='No_Of_Casualities',
                    hue=cas_count.index.get_level_values('urban_or_rural_area'), data=cas_count.reset_index(),
                    palette='Set1', ax=ax)
        # Remove the outline
        sns.despine()
        plt.xticks(rotation=45)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Number of Casualties", fontsize=12)
        plt.legend(title="Urban/Rural Area", loc='upper right')
        plt.tight_layout()
        st.pyplot(fig)
    # Create a FacetGrid
    st.markdown(f"<h2 style='text-align: left; font-size: 20px; color: #008080;'>"
                f"Number of vehicles involved as per accident severity</h2>",
                unsafe_allow_html=True)
    grid = sns.FacetGrid(data=df, col='accident_severity', height=4, aspect=1, sharey=False)

    # Map the count plot onto the grid
    grid.map(sns.countplot, 'number_of_vehicles', palette=['black', 'brown', 'orange'])

    # Display the plot
    fig = grid.fig
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        # Plot pie chart accidents by days of the week
        st.markdown(f"<h2 style='text-align: left; font-size: 20px; color: #008080;'>"
                    f"Accidents by Days of the Week</h2>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        df['day_of_week'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        plt.axis('equal')
        st.pyplot(fig)
    with col2:
        # Plot bar chart accidents by accident severity
        st.markdown(f"<h2 style='text-align: left; font-size: 20px; color: #008080;'>"
                    f"Accident Severity Distribution</h2>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        severity_counts = df['accident_severity'].value_counts()
        severity_counts.plot(kind='barh', color='orange')
        plt.xlabel("Number of Accidents", fontsize=12)
        plt.ylabel("Accident Severity", fontsize=12)
        # Remove the outline
        sns.despine()
        plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add grid lines for better readability
        st.pyplot(fig)
    # KPI 1
    # Calculate number of accidents
    accident_count = df.shape[0]
    # Display as a KPI
    st.sidebar.markdown(f"<h2 style='text-align: left; font-size: 20px; color: #008080;'>Total Accidents</h2>",
                        unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='text-align: left; font-size: 18px; color: #FF5733;'>{accident_count}</p>",
                        unsafe_allow_html=True)

    # KPI 2
    # Calculate total number of casualties
    total_casualties = df['number_of_casualties'].sum()
    # Display as a KPI
    st.sidebar.markdown(f"<h2 style='text-align: left; font-size: 20px; color: #008080;'>Total Casualties</h2>",
                        unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='text-align: left; font-size: 18px; color: #FF5733;'>{total_casualties}</p>",
                        unsafe_allow_html=True)

    # KPI 3
    # Calculate total number of vehicles involved
    total_vehicles = df['number_of_vehicles'].sum()

    # Display as a KPI
    st.sidebar.markdown(f"<h2 style='text-align: left; font-size: 20px; color: #008080;'>Total Vehicles Involved</h2>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p style='text-align: left; font-size: 18px; color: #FF5733;'>{total_vehicles}</p>", unsafe_allow_html=True)

# if you have selected prediction button this section will be displayed
elif selection == "Prediction":
    # pickle file is imported using load function
    load_model = pickle.load(open('trail.pkl', 'rb'))

    # Define the classifier names and corresponding file paths
    classifiers = {
        'Extra Trees': 'extratree.csv',
        'Random Forest': 'randomforest.csv',
        'KNN': 'knn.csv'
    }

    # Create a select box to choose a classifier
    classifier_name = st.sidebar.selectbox("Choose a classifier", list(classifiers.keys()))

    # Check if a classifier is selected
    if classifier_name:
        # Read the corresponding accuracy file
        df = pd.read_csv(classifiers[classifier_name])

        # Extract the accuracy value
        accuracy_final = df.values[0][0]
        # Display the result based on the selected classifier
        st.sidebar.write(f"The selected classifier is {classifier_name} with an Accuracy of {accuracy_final}")
    else:
        # Display a message if no classifier is selected
        st.sidebar.warning("Please select a classifier")

        # Function to reshape input data
    def accident_severity(input_data):
        # Convert input data into Numpy array
        input_data_numpy_array = np.asarray(input_data)
        # Reshape input data to the format expected by the model
        input_data_reshaped = input_data_numpy_array.reshape(1, -1)
        # Make predictions using the loaded model
        prediction = load_model.predict(input_data_reshaped)
        # Print prediction
        print(prediction)
        # Map the prediction value to text
        if prediction == 1:
            return "Fatal Accident"
        elif prediction == 2:
            return "Serious Accident"
        elif prediction == 3:
            return "Slight Accident"

    # Function to display input fields
    def main():
        # title for app
        st.title('Accident Severity')
        col1, col2, col3 = st.columns(3)
        # get the inputs
        with col1:
            number_of_vehicles = st.text_input("Number of Vehicles")
            number_of_casualties = st.text_input("Number of Casualties")
            day = st.text_input("Day")
            month = st.text_input("Month")
            year = st.text_input("Year")

        with col2:
            speed_limit = st.text_input("Speed Limit")
            pedestrian_crossing_physical_facilities = st.selectbox("Pedestrian Crossing",
                                                                   ['No physical crossing facilities within 50 metres',
                                                                    'Zebra',
                                                                    'Pelican, puffin, toucan or similar non-junction pedestrian light crossing',
                                                                    'Pedestrian phase at traffic signal junction',
                                                                    'Footbridge or subway', 'Central refuge',
                                                                    'unknown (self reported)'])
            light_conditions = st.selectbox("Light Conditions", ['Daylight', 'Darkness - lights lit',
                                                                 'Darkness - lights unlit', 'Darkness - no lighting',
                                                                 'Darkness - lighting unknown'])
            weather_conditions = st.selectbox("Weather Conditions", ['Fine no high winds', 'Raining no high winds',
                                                                     'Snowing no high winds', 'Fine + high winds',
                                                                     'Raining + high winds', 'Snowing + high winds',
                                                                     'Fog or mist', 'Other', 'Unknown'])
            road_surface_conditions = st.selectbox("Road Surface Conditions", ['Dry', 'Wet or damp', 'Snow',
                                                                               'Frost or ice', 'Flood over 3cm. deep',
                                                                               'Oil or diesel', 'Mud'])
        with col3:
            did_police_officer_attend_scene_of_accident = st.selectbox("Police Attended Collision Scene", ['Yes', 'No',
                                                                                                           'No, Self completion'])
            day_of_week = st.selectbox("Day Of Week", ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                                                       'Saturday'])
            road_type = st.selectbox("Road Type", ['Roundabout', 'One way street', 'Dual carriageway',
                                                   'Single carriageway', 'Slip road', 'Unknown'])
            police_force = st.selectbox("Police Force", ['Metropolitan Police', 'Cumbria', 'Lancashire', 'Merseyside',
                                                         'Greater Manchester', 'Cheshire', 'Northumbria', 'Durham',
                                                         'North Yorkshire', 'West Yorkshire', 'South Yorkshire',
                                                         'Humberside', 'Cleveland', 'West Midlands', 'Staffordshire',
                                                         'West Mercia', 'Warwickshire', 'Derbyshire', 'Nottinghamshire',
                                                         'Lincolnshire', 'Leicestershire', 'Northamptonshire',
                                                         'Cambridgeshire', 'Norfolk', 'Suffolk', 'Bedfordshire',
                                                         'Hertfordshire', 'Essex', 'Thames Valley', 'Hampshire',
                                                         'Surrey', 'Kent', 'Sussex', 'City of London',
                                                         'Devon and Cornwall', 'Avon and Somerset', 'Gloucestershire',
                                                         'Wiltshire', 'Dorset', 'North Wales', 'Gwent', 'South Wales',
                                                         'Dyfed-Powys', 'Northern', 'Grampian', 'Tayside', 'Fife',
                                                         'Lothian and Borders', 'Central', 'Strathclyde',
                                                         'Dumfries and Galloway', 'Police Scotland'])

# mapping the values
        if road_type == 'Roundabout':
            road_type = 1
        elif road_type == 'One way street':
            road_type = 2
        elif road_type == 'Dual carriageway':
            road_type = 3
        elif road_type == 'Single carriageway':
            road_type = 6
        elif road_type == 'Slip road':
            road_type = 7
        elif road_type == 'Unknown':
            road_type = 9

        if day_of_week == 'Sunday':
            day_of_week = 1
        elif day_of_week == 'Monday':
            day_of_week = 2
        elif day_of_week == 'Tuesday':
            day_of_week = 3
        elif day_of_week == 'Wednesday':
            day_of_week = 4
        elif day_of_week == 'Thursday':
            day_of_week = 5
        elif day_of_week == 'Friday':
            day_of_week = 6
        elif day_of_week == 'Saturday':
            day_of_week = 7

        if police_force == 'Metropolitan Police':
            police_force = 1
        elif police_force == 'Cumbria':
            police_force = 3
        elif police_force == 'Lancashire':
            police_force = 4
        elif police_force == 'Greater Manchester':
            police_force = 6
        elif police_force == 'Cheshire':
            police_force = 7
        elif police_force == 'Northumbria':
            police_force = 10
        elif police_force == 'Durham':
            police_force = 11
        elif police_force == 'North Yorkshire':
            police_force = 12
        elif police_force == 'West Yorkshire':
            police_force = 13
        elif police_force == 'South Yorkshire':
            police_force = 14
        elif police_force == 'Humberside':
            police_force = 16
        elif police_force == 'Cleveland':
            police_force = 17
        elif police_force == 'West Midlands':
            police_force = 20
        elif police_force == 'Staffordshire':
            police_force = 21
        elif police_force == 'West Mercia':
            police_force = 22
        elif police_force == 'Warwickshire':
            police_force = 23
        elif police_force == 'Derbyshire':
            police_force = 30
        elif police_force == 'Nottinghamshire':
            police_force = 31
        elif police_force == 'Lincolnshire':
            police_force = 32
        elif police_force == 'Leicestershire':
            police_force = 33
        elif police_force == 'Northamptonshire':
            police_force = 34
        elif police_force == 'Cambridgeshire':
            police_force = 35
        elif police_force == 'Norfolk':
            police_force = 36
        elif police_force == 'Suffolk':
            police_force = 37
        elif police_force == 'Bedfordshire':
            police_force = 40
        elif police_force == 'Hertfordshire':
            police_force = 41
        elif police_force == 'Essex':
            police_force = 42
        elif police_force == 'Thames Valley':
            police_force = 43
        elif police_force == 'Hampshire':
            police_force = 44
        elif police_force == 'Surrey':
            police_force = 45
        elif police_force == 'Kent':
            police_force = 46
        elif police_force == 'Sussex':
            police_force = 47
        elif police_force == 'City of London':
            police_force = 48
        elif police_force == 'Devon and Cornwall':
            police_force = 50
        elif police_force == 'Avon and Somerset':
            police_force = 52
        elif police_force == 'Gloucestershire':
            police_force = 53
        elif police_force == 'Wiltshire':
            police_force = 54
        elif police_force == 'Dorset':
            police_force = 55
        elif police_force == 'North Wales':
            police_force = 60
        elif police_force == 'Gwent':
            police_force = 61
        elif police_force == 'South Wales':
            police_force = 62
        elif police_force == 'Dyfed-Powys':
            police_force = 63
        elif police_force == 'Northern':
            police_force = 91
        elif police_force == 'Grampian':
            police_force = 92
        elif police_force == 'Tayside':
            police_force = 93
        elif police_force == 'Fife':
            police_force = 94
        elif police_force == 'Lothian and Borders':
            police_force = 95
        elif police_force == 'Central':
            police_force = 96
        elif police_force == 'Strathclyde':
            police_force = 97
        elif police_force == 'Dumfries and Galloway':
            police_force = 98
        elif police_force == 'Police Scotland':
            police_force = 99

        if pedestrian_crossing_physical_facilities == 'No physical crossing facilities within 50 metres':
            pedestrian_crossing_physical_facilities = 0
        elif pedestrian_crossing_physical_facilities == 'Zebra':
            pedestrian_crossing_physical_facilities = 1
        elif pedestrian_crossing_physical_facilities == 'Pelican, puffin, toucan or similar non-junction pedestrian light crossing':
            pedestrian_crossing_physical_facilities = 4
        elif pedestrian_crossing_physical_facilities == 'Pedestrian phase at traffic signal junction':
            pedestrian_crossing_physical_facilities = 5
        elif pedestrian_crossing_physical_facilities == 'Footbridge or subway':
            pedestrian_crossing_physical_facilities = 7
        elif pedestrian_crossing_physical_facilities == 'Central refuge':
            pedestrian_crossing_physical_facilities = 8
        elif pedestrian_crossing_physical_facilities == 'unknown (self reported)':
            pedestrian_crossing_physical_facilities = 9

        if light_conditions == 'Daylight':
            light_conditions = 1
        elif light_conditions == 'Darkness - lights lit':
            light_conditions = 4
        elif light_conditions == 'Darkness - lights unlit':
            light_conditions = 5
        elif light_conditions == 'Darkness - no lighting':
            light_conditions = 6
        elif light_conditions == 'Darkness - lighting unknown':
            light_conditions = 7

        if weather_conditions == 'Fine no high winds':
            weather_conditions = 1
        elif weather_conditions == 'Raining no high winds':
            weather_conditions = 2
        elif weather_conditions == 'Snowing no high winds':
            weather_conditions = 3
        elif weather_conditions == 'Fine + high winds':
            weather_conditions = 4
        elif weather_conditions == 'Raining + high winds':
            weather_conditions = 5
        elif weather_conditions == 'Snowing + high winds':
            weather_conditions = 6
        elif weather_conditions == 'Fog or mist':
            weather_conditions = 7
        elif weather_conditions == 'Other':
            weather_conditions = 8
        elif weather_conditions == 'Unknown':
            weather_conditions = 9

        if road_surface_conditions == 'Dry':
            road_surface_conditions = 1
        elif road_surface_conditions == 'Wet or damp':
            road_surface_conditions = 2
        elif road_surface_conditions == 'Snow':
            road_surface_conditions = 3
        elif road_surface_conditions == 'Frost or ice':
            road_surface_conditions = 4
        elif road_surface_conditions == 'Flood over 3cm. deep':
            road_surface_conditions = 5
        elif road_surface_conditions == 'Oil or diesel':
            road_surface_conditions = 6
        elif road_surface_conditions == 'Mud':
            road_surface_conditions = 7

        if did_police_officer_attend_scene_of_accident == 'Yes':
            did_police_officer_attend_scene_of_accident = 1
        elif did_police_officer_attend_scene_of_accident == 'No':
            did_police_officer_attend_scene_of_accident = 2
        elif did_police_officer_attend_scene_of_accident == 'No, Self completion':
            did_police_officer_attend_scene_of_accident = 3
        # code for prediction
        severity = ''

        # creating button for prediction
        if st.button('ACCIDENT_SEVERITY'):
            severity = accident_severity([police_force, number_of_vehicles, number_of_casualties, day_of_week,
                                          road_type, speed_limit, pedestrian_crossing_physical_facilities,
                                          light_conditions, weather_conditions, road_surface_conditions,
                                          did_police_officer_attend_scene_of_accident, day, month, year])
        st.success(severity)


    if __name__ == '__main__':
        main()

