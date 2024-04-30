import streamlit as st 
from datetime import datetime, date, timedelta
import pandas as pd
from Home import face_rec
import random
# st.set_page_config(page_title='Reporting',layout='wide')
st.subheader('Reporting')


# Retrive logs data and show in Report.py
# extract data from redis list
name = 'attendance:logs'
def load_logs(name,end=-1):
    logs_list = face_rec.r.lrange(name,start=0,end=end) # extract all data from the redis database
    return logs_list

# tabs to show the info
tab1, tab2 = st.tabs(['Registered Data', 'Attendance Report'])
tab4, tab5, tab6 = st.tabs(['Daily','Weekly', 'Monthly'])
tab7, tab8, tab9 = st.tabs(['Semestral','Tardy', 'Perfect Attendance'])

def default_tab():
    redis_face_db = face_rec.retrive_data(name='academy:register')
    result = ['Name','Role','Course & Level','Section','School Year','Address','Contact Number','Email']
    return st.dataframe(redis_face_db[result])


def content_display():
    redis_face_db = face_rec.retrive_data(name='academy:register')
    result = ['Name','Role','Course & Level','Section','School Year','Address','Contact Number','Email', 'Professor']
    
    if redis_face_db[result]['Professor'][0] != '':
        st.write("Teacher : "+redis_face_db[result]['Professor'][0])
        st.write("Year Level : 2024")
        st.write("Course : "+redis_face_db[result]['Course & Level'][0])
        st.write("Section : "+redis_face_db[result]['Section'][0])
        st.write("\n")
        st.write("\n")
    # return redis_face_db[result]

def get_week_dates(selected_date):
    # Calculate the start date (Monday) of the week containing the selected date
    start_date = selected_date + timedelta(days=selected_date.weekday())
    
    # Calculate the end date (Sunday) of the week containing the selected date
    end_date = start_date + timedelta(days=6)
    
    # Generate a list of dates for the week
    week_dates = [start_date + timedelta(days=i) for i in range(7)]
    
    return week_dates


def generate_weekly_report(start_date):
    # Generate dates for the week starting from the given start_date
    # start_date = start_date + timedelta(days=start_date.weekday())
    dates = [start_date + timedelta(days=i) for i in range(7)]
    
    return dates

def generate_monthly_attendance_table(logs_list, users, start_date, end_date, attendance_records):
    
    num_weeks = (end_date - start_date).days // 7 + 1
    data = {'User': users}
    for week_num in range(1, num_weeks + 1):
        data[f'Week {week_num}'] = [''] * len(users)
        
    for record in attendance_records:
        user = record['Student Name']
        if user in users:
            for week_num in range(1, num_weeks + 1):
                start_week = start_date + timedelta(days=(week_num - 1) * 7)
                end_week = min(start_date + timedelta(days=week_num * 7 - 1), end_date)
                attendance_status = get_weekly_attendance_status(logs_list, record, start_week, end_week)
                data[f'Week {week_num}'][users.index(user)] = attendance_status

    df = pd.DataFrame(data)
    # st.write(df.to_html(index=False, escape=False), unsafe_allow_html=True)
    st.write(df)

def get_weekly_attendance_status(logs_list, record, start_date, end_date):
    # Calculate the attendance status for the specified week
    # st.write(record)
    attendance_status = ''
    for day in pd.date_range(start=start_date, end=end_date):
        
        day_str = day.strftime('%Y-%m-%d')
        if day_str in record:
            if record[day_str] != "":
                attendance_status += day.strftime('%a')+' - '+record[day_str]+' | '

    attendance_status = attendance_status[:-3]
    # attendance_status = attendance_status.rstrip(' | ')  # Remove trailing slash
    
    return attendance_status


def generate_monthly_days():
    # Get the current date
    current_date = date.today()

    # Get the first day of the current month
    first_day_of_month = current_date.replace(day=1)

    # Calculate the last day of the current month
    next_month = first_day_of_month.replace(month=first_day_of_month.month + 1)
    last_day_of_month = next_month - timedelta(days=1)

    # Generate all dates of the current month
    dates_of_month = [first_day_of_month + timedelta(days=i) for i in range((last_day_of_month - first_day_of_month).days + 1)]

    return dates_of_month


def daily_views(logs_list):
    # Step 1: Convert the logs that in list of bytes into list of string
    convert_byte_to_string = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_byte_to_string, logs_list))

    # Step 2: Split string by @ and create nested list
    split_string = lambda x: x.split('|')
    logs_nested_list = list(map(split_string, logs_list_string))
    
    # Convert nested list info into dataframe
    logs_df = pd.DataFrame(logs_nested_list, columns= ['Name','Role','Course & Level','Section','School Year','Address','Contact Number','Email','Professor','Timestamp'])
    
    # Step 3: Time base Analysis or Report Time-In Time-out
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date

    # Step 3.1: Calculate Report Time-In Time-out
    # In Time: At which person is first detected in that day (min Timestamp of the date)
    # Out Time: At which person is last detected in that day (max Timestamp of the date)
    
    report_df = logs_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'), # In Time
        Out_time = pd.NamedAgg('Timestamp','max') # Out Time
    ).reset_index()
    report_df['Address'] = logs_df['Address']
    report_df['Email'] = logs_df['Email']
    report_df['Contact Number'] = logs_df['Contact Number']
    report_df['Course & Level'] = logs_df['Course & Level']+logs_df['Section']
    report_df['Section'] = logs_df['Section']
    report_df['Professor'] = logs_df['Professor']
    
    report_df['In Time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out Time'] = pd.to_datetime(report_df['Out_time'])

    report_df['Duration'] = report_df['Out Time'] - report_df['In Time']

    # Step 4: Marking Person is Present or Absent
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name','Role']].drop_duplicates().values.tolist()

    date_name_rol_zip_df = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_rol_zip_df.append([dt, name, role])

    date_name_rol_zip_df = pd.DataFrame(date_name_rol_zip_df, columns=['Date','Name','Role'])
    
    # lef join with report_df
    date_name_rol_zip_df = pd.merge(date_name_rol_zip_df, report_df, how='left',on=['Date','Name','Role'])
    date_name_rol_zip_df = date_name_rol_zip_df[(date_name_rol_zip_df['Date'] == datetime.today().date()) & (date_name_rol_zip_df['Role'] == "Student")]
    date_name_rol_zip_df['Duration_seconds'] = date_name_rol_zip_df['Duration'].dt.seconds
    date_name_rol_zip_df['Duration_hours'] = date_name_rol_zip_df['Duration_seconds'] / (60*60)
    
    
    ress = pd.DataFrame(date_name_rol_zip_df, columns=['Date','Role', 'Name', 'In Time', 'Out Time', 'Address', 'Course & Level', 'Contact Number', 'Email', 'Timestamp', 'Section'])
    processed_logs_df = ress.drop_duplicates(subset=['Name','Role', 'Date'], keep='first')
    processed_logs_df.reset_index(drop=True, inplace=True)
    final_df = processed_logs_df[['Role', 'Name', 'In Time', 'Out Time', 'Address', 'Course & Level', 'Contact Number', 'Email']]
    
    return final_df
    


def weekly_views(logs_list, first_d, last_d):
    convert_byte_to_string = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_byte_to_string, logs_list))

    split_string = lambda x: x.split('|')
    logs_nested_list = list(map(split_string, logs_list_string))
    logs_df = pd.DataFrame(logs_nested_list, columns= ['Name','Role','Course & Level','Section','School Year','Address','Contact Number','Email','Professor','Timestamp'])

    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date
    report_df = logs_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'), # In Time
        Out_time = pd.NamedAgg('Timestamp','max') # Out Time
    ).reset_index()
    report_df['In Time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out Time'] = pd.to_datetime(report_df['Out_time'])

    report_df['Duration'] = report_df['Out Time'] - report_df['In Time']
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name','Role']].drop_duplicates().values.tolist()

    date_name_rol_zip_df = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_rol_zip_df.append([dt, name, role])

    date_name_rol_zip_df = pd.DataFrame(date_name_rol_zip_df, columns=['Date','Name','Role'])

    date_name_rol_zip_df = pd.merge(date_name_rol_zip_df, report_df, how='left',on=['Date','Name','Role'])
    date_name_rol_zip_df = date_name_rol_zip_df[(date_name_rol_zip_df['Date'] >= first_d) & (date_name_rol_zip_df['Date'] <= last_d) & (date_name_rol_zip_df['Role'] == "Student")]

    date_name_rol_zip_df['Duration_seconds'] = date_name_rol_zip_df['Duration'].dt.seconds
    date_name_rol_zip_df['Duration_hours'] = date_name_rol_zip_df['Duration_seconds'] / (60*60)

    def status_marker(x):

        if pd.Series(x).isnull().all():
            return 'Absent'        
        elif x >= 0 and x < 1:
            return 'Late'
        elif x >= 2 and x <= 10:
            return 'Present'
        
    date_name_rol_zip_df['Status'] = date_name_rol_zip_df['Duration_hours'].apply(status_marker)
    date_name_rol_zip_df['Date'] = pd.to_datetime(date_name_rol_zip_df['Date'])
    date_name_rol_zip_df.set_index('Date', inplace=True)

    weekly_report = date_name_rol_zip_df.groupby(['Name','Role', pd.Grouper(freq='W-Mon')]).agg(
        Total_Present=pd.NamedAgg(column='Status', aggfunc=lambda x: (x == 'Present').sum()),
        Total_Absent=pd.NamedAgg(column='Status', aggfunc=lambda x: (x == 'Absent').sum())
            ).reset_index()

    date_name_rol_zip_df.reset_index(inplace=True)
    
    date_name_rol_zip_df['date_content'] = date_name_rol_zip_df.groupby('Name')['Date'].transform(lambda x: ','.join(x.astype(str).unique()))
    date_name_rol_zip_df['stat'] = date_name_rol_zip_df.groupby('Name')['Status'].transform(lambda x: ','.join(x.astype(str)))
    date_name_rol_zip_df = date_name_rol_zip_df.drop_duplicates(subset='Name', keep='first')

    return pd.DataFrame(date_name_rol_zip_df)


def monthly_views(logs_list, first_d, last_d):
    convert_byte_to_string = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_byte_to_string, logs_list))

    split_string = lambda x: x.split('|')
    logs_nested_list = list(map(split_string, logs_list_string))
    logs_df = pd.DataFrame(logs_nested_list, columns= ['Name','Role','Course & Level','Section','School Year','Address','Contact Number','Email','Professor','Timestamp'])

    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date
    report_df = logs_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'), # In Time
        Out_time = pd.NamedAgg('Timestamp','max') # Out Time
    ).reset_index()
    report_df['In Time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out Time'] = pd.to_datetime(report_df['Out_time'])

    report_df['Duration'] = report_df['Out Time'] - report_df['In Time']
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name','Role']].drop_duplicates().values.tolist()

    date_name_rol_zip_df = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_rol_zip_df.append([dt, name, role])

    date_name_rol_zip_df = pd.DataFrame(date_name_rol_zip_df, columns=['Date','Name','Role'])

    date_name_rol_zip_df = pd.merge(date_name_rol_zip_df, report_df, how='left',on=['Date','Name','Role'])
    
    date_name_rol_zip_df['Duration_seconds'] = date_name_rol_zip_df['Duration'].dt.seconds
    date_name_rol_zip_df['Duration_hours'] = date_name_rol_zip_df['Duration_seconds'] / (60*60)

    def status_marker(x):

        if pd.Series(x).isnull().all():
            return 'Absent'        
        elif x >= 0 and x < 1:
            return 'Late'
        elif x >= 2 and x <= 10:
            return 'Present'
        
    date_name_rol_zip_df['Status'] = date_name_rol_zip_df['Duration_hours'].apply(status_marker)
    date_name_rol_zip_df['Date'] = pd.to_datetime(date_name_rol_zip_df['Date'])
    date_name_rol_zip_df.set_index('Date', inplace=True)

    weekly_report = date_name_rol_zip_df.groupby(['Name','Role', pd.Grouper(freq='W-Mon')]).agg(
        Total_Present=pd.NamedAgg(column='Status', aggfunc=lambda x: (x == 'Present').sum()),
        Total_Absent=pd.NamedAgg(column='Status', aggfunc=lambda x: (x == 'Absent').sum())
            ).reset_index()

    date_name_rol_zip_df.reset_index(inplace=True)
    
    date_name_rol_zip_df['date_content'] = date_name_rol_zip_df.groupby('Name')['Date'].transform(lambda x: ','.join(x.astype(str).unique()))
    date_name_rol_zip_df['stat'] = date_name_rol_zip_df.groupby('Name')['Status'].transform(lambda x: ','.join(x.astype(str)))
    date_name_rol_zip_df = date_name_rol_zip_df.drop_duplicates(subset='Name', keep='first')

    return pd.DataFrame(date_name_rol_zip_df)



def tardy_views(logs_list):
    # Step 1: Convert the logs that in list of bytes into list of string
    convert_byte_to_string = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_byte_to_string, logs_list))

    # Step 2: Split string by @ and create nested list
    split_string = lambda x: x.split('|')
    logs_nested_list = list(map(split_string, logs_list_string))
    # Convert nested list info into dataframe
    logs_df = pd.DataFrame(logs_nested_list, columns= ['Name','Role','Course & Level','Section','School Year','Address','Contact Number','Email','Professor','Timestamp'])

    # Step 3: Time base Analysis or Report Time-In Time-out
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date

    # Step 3.1: Calculate Report Time-In Time-out
    # In Time: At which person is first detected in that day (min Timestamp of the date)
    # Out Time: At which person is last detected in that day (max Timestamp of the date)
    
    report_df = logs_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'), # In Time
        Out_time = pd.NamedAgg('Timestamp','max') # Out Time
    ).reset_index()
    report_df['Address'] = logs_df['Address']
    report_df['Email'] = logs_df['Email']
    report_df['Contact Number'] = logs_df['Contact Number']
    report_df['Course & Level'] = logs_df['Course & Level']+logs_df['Section']
    report_df['Section'] = logs_df['Section']
    report_df['Professor'] = logs_df['Professor']
    report_df['In Time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out Time'] = pd.to_datetime(report_df['Out_time'])

    report_df['Duration'] = report_df['Out Time'] - report_df['In Time']

    # Step 4: Marking Person is Present or Absent
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name','Role']].drop_duplicates().values.tolist()

    date_name_rol_zip_df = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_rol_zip_df.append([dt, name, role])

    date_name_rol_zip_df = pd.DataFrame(date_name_rol_zip_df, columns=['Date','Name','Role'])

    # lef join with report_df
    date_name_rol_zip_df = pd.merge(date_name_rol_zip_df, report_df, how='left',on=['Date','Name','Role'])
    date_name_rol_zip_df['Duration_seconds'] = date_name_rol_zip_df['Duration'].dt.seconds
    date_name_rol_zip_df['Duration_hours'] = date_name_rol_zip_df['Duration_seconds'] / (60*60)
    
    def status_marker(x):

        if pd.Series(x).isnull().all():
            return 'Absent'
        
        elif x >= 0 and x < 1:
            return 'Late'
        
        elif x >= 2 and x <= 10:
            return 'Present'
        
    date_name_rol_zip_df['Status'] = date_name_rol_zip_df['Duration_hours'].apply(status_marker)

# Convert 'Date' column to datetime if it's not already
    date_name_rol_zip_df['Date'] = pd.to_datetime(date_name_rol_zip_df['Date'])

# Set 'Date' as the index
    date_name_rol_zip_df.set_index('Date', inplace=True)

# Step 6: Monthly Report
    monthly_report = date_name_rol_zip_df.groupby(['Name','Role', pd.Grouper(freq='M')]).agg(
    Total_Present=pd.NamedAgg(column='Status', aggfunc=lambda x: (x == 'Present').sum()),
    Total_Absent=pd.NamedAgg(column='Status', aggfunc=lambda x: (x == 'Absent').sum())
        ).reset_index()


    date_name_rol_zip_df.reset_index(inplace=True)
    monthly_report[['Date', 'Name', 'Role', 'Total_Present', 'Total_Absent']].to_csv("Monthly_Attendence_Report.csv",index=False)
    df = pd.DataFrame(monthly_report, columns=['Name', 'Total_Absent'])
    st.write(df)



def perfect_attendance_view(logs_list):
    convert_byte_to_string = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_byte_to_string, logs_list))

    # Step 2: Split string by @ and create nested list
    split_string = lambda x: x.split('|')
    logs_nested_list = list(map(split_string, logs_list_string))
    
    logs_df = pd.DataFrame(logs_nested_list, columns= ['Name','Role','Course & Level','Section','School Year','Address','Contact Number','Email','Professor','Timestamp'])

    # Step 3: Time base Analysis or Report Time-In Time-out
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date

    
    report_df = logs_df.groupby(by=['Date','Name','Role']).agg(
        In_time = pd.NamedAgg('Timestamp','min'), # In Time
        Out_time = pd.NamedAgg('Timestamp','max') # Out Time
    ).reset_index()
    report_df['Address'] = logs_df['Address']
    report_df['Email'] = logs_df['Email']
    report_df['Contact Number'] = logs_df['Contact Number']
    report_df['Course & Level'] = logs_df['Course & Level']+logs_df['Section']
    report_df['Section'] = logs_df['Section']
    report_df['Professor'] = logs_df['Professor']
    
    report_df['In Time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out Time'] = pd.to_datetime(report_df['Out_time'])

    report_df['Duration'] = report_df['Out Time'] - report_df['In Time']

    # Step 4: Marking Person is Present or Absent
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name','Role']].drop_duplicates().values.tolist()

    date_name_rol_zip_df = []
    for dt in all_dates:
        for name, role in name_role:
            date_name_rol_zip_df.append([dt, name, role])

    date_name_rol_zip_df = pd.DataFrame(date_name_rol_zip_df, columns=['Date','Name','Role'])

    # lef join with report_df
    date_name_rol_zip_df = pd.merge(date_name_rol_zip_df, report_df, how='left',on=['Date','Name','Role'])
    
    date_name_rol_zip_df['Duration_seconds'] = date_name_rol_zip_df['Duration'].dt.seconds
    date_name_rol_zip_df['Duration_hours'] = date_name_rol_zip_df['Duration_seconds'] / (60*60)
    
    def status_marker(x):

        if pd.Series(x).isnull().all():
            return 'Absent'
        
        elif x >= 0 and x < 1:
            return 'Late'
        
        elif x >= 2 and x <= 10:
            return 'Present'
        
    date_name_rol_zip_df['Status'] = date_name_rol_zip_df['Duration_hours'].apply(status_marker)

# Convert 'Date' column to datetime if it's not already
    date_name_rol_zip_df['Date'] = pd.to_datetime(date_name_rol_zip_df['Date'])

# Set 'Date' as the index
    date_name_rol_zip_df.set_index('Date', inplace=True)

# Step 6: Monthly Report
    monthly_report = date_name_rol_zip_df.groupby(['Name','Role', pd.Grouper(freq='M')]).agg(
    Total_Present=pd.NamedAgg(column='Status', aggfunc=lambda x: (x == 'Present').sum()),
    Total_Absent=pd.NamedAgg(column='Status', aggfunc=lambda x: (x == 'Absent').sum())
        ).reset_index()

# Reset the index in the original dataframe
    date_name_rol_zip_df.reset_index(inplace=True)

# Display the monthly report
    monthly_report[['Date', 'Name', 'Role', 'Total_Present', 'Total_Absent']].to_csv("Monthly_Attendence_Report.csv",index=False)
    st.dataframe(monthly_report, column_order=['Date', 'Name', 'Role', 'Total_Present', 'Total_Absent'])


def count_present_days(month_attendance_records):
    """
    Count the total number of 'Present' days for each student in the monthly attendance records.
    """
    present_counts = {}

    # Iterate over each row in the DataFrame
    for index, row in month_attendance_records.iterrows():
        student_name = row['Student Name']
        present_count = sum(1 for day, status in row.items() if status == 'Present' and day != 'Student Name')
        present_counts[student_name] = present_count
        
    return present_counts


with tab1:
    logs_list = load_logs(name=name)

    if st.button('Refresh Data'):
        with st.spinner('Retriving Data from Redis DB ...'):   
            print('reload')

    content_display()
    with tab4:
        st.dataframe(daily_views(logs_list))
        
    with tab5:
        # Date input to specify the start of the week
        start_date = st.date_input("Select a start date of the week")
        if start_date:
            result = generate_weekly_report(start_date)
            res = weekly_views(logs_list, result[0], result[-1])
            res = pd.DataFrame(res)

            days_list = [i.strftime("%A") for i in result]

            dynamic_key = {'Student Name': [], **{day: [] for day in days_list}}
            users= []
            for st_user in res['Name']:
                users.append(st_user)
            
            attendance_records = []
            for user, dts, stas in zip(users, res['date_content'], res['stat']):
                date_conts = dts.split(',')
                attnds = stas.split(',')
                record = {'Student Name': user}
                for day in days_list:
                    record[day] = ''  # Initialize all days as blank
                    for date_cont, attnd in zip(date_conts, attnds):
                        date_cnt = (datetime.strptime(date_cont, "%Y-%m-%d")).strftime("%A")
                        if day == date_cnt:
                            if attnd == 'Late':
                                record[day] ="Present"
                            if attnd != 'Late':
                                record[day] = attnd

                attendance_records.append(record)
            for user_attendance in attendance_records:
                user = user_attendance['Student Name']
                if user in users:
                    dynamic_key['Student Name'].append(user)
                    for day in days_list:
                        dynamic_key[day].append(user_attendance[day])

            df = pd.DataFrame(dynamic_key)
            st.write(df)
            
    with tab6:
        month_result = generate_monthly_days()
        resu = monthly_views(logs_list, month_result[0], month_result[-1])
        resu = pd.DataFrame(resu)

        month_days_list = [i.strftime("%Y-%m-%d") for i in month_result]

        monthly_dynamic_key = {'Student Name': [], **{day: [] for day in month_result}}
        monthly_users= []
        for st_user in resu['Name']:
            monthly_users.append(st_user)
        
        month_attendance_records = []
        for user, dts, stas in zip(monthly_users, resu['date_content'], resu['stat']):
            date_conts = dts.split(',')
            attnds = stas.split(',')
            record = {'Student Name': user}
            for day in month_days_list:
                record[day] = ''  # Initialize all days as blank
                for date_cont, attnd in zip(date_conts, attnds):
                    date_cnt = (datetime.strptime(date_cont, "%Y-%m-%d")).strftime("%Y-%m-%d")
                    if day == date_cnt:
                        if attnd == 'Late':
                            record[day] ="Present"
                        if attnd != 'Late':
                            record[day] = attnd

            month_attendance_records.append(record)
        generate_monthly_attendance_table(logs_list, monthly_users, month_result[0], month_result[-1], month_attendance_records)

    with tab7:
        default_tab()
    with tab8:
        tardy_views(logs_list)
    with tab9:
        month_result = generate_monthly_days()
        resu = monthly_views(logs_list, month_result[0], month_result[-1])
        resu = pd.DataFrame(resu)
        # st.write(resu)
        month_days_list = [i.strftime("%Y-%m-%d") for i in month_result]

        monthly_dynamic_key = {'Student Name': [], **{day: [] for day in month_result}}
        monthly_users= []
        for st_user in resu['Name']:
            monthly_users.append(st_user)
        
        month_attendance_records = []
        for user, dts, stas in zip(monthly_users, resu['date_content'], resu['stat']):
            date_conts = dts.split(',')
            attnds = stas.split(',')
            record = {'Student Name': user}
            for day in month_days_list:
                record[day] = ''  # Initialize all days as blank
                for date_cont, attnd in zip(date_conts, attnds):
                    date_cnt = (datetime.strptime(date_cont, "%Y-%m-%d")).strftime("%Y-%m-%d")
                    if day == date_cnt:
                        if attnd == 'Late':
                            record[day] ='Present'
                        if attnd != 'Late':
                            record[day] =attnd
                        record['total_scol_days'] = '100'

            month_attendance_records.append(record)
        resu = pd.DataFrame(month_attendance_records)
        present_counts = count_present_days(resu)
        df = pd.DataFrame(list(present_counts.items()), columns=['Student Name', 'Number of School Days Present'])
        df['Number of School Days - Semestral'] = 100
        df = df[['Student Name', 'Number of School Days - Semestral', 'Number of School Days Present']]
        st.write(df)

with tab2:
    st.subheader('Registered Data')
    

