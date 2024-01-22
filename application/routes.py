from application import app
from flask import render_template, url_for, request, session, redirect
import pandas as pd
import matplotlib.pyplot as plt
import json
import plotly
import plotly_express as px
import numpy as np
import plotly.graph_objects as go

df = None
target = ""
col_names = []
numerical_cols = []
non_numeric_cols = []
best = None



@app.route("/")
def home():
    return render_template("home.html", title="Home")




@app.route("/select_target", methods=["GET", "POST"])
def select_target():
    csv_file = request.files['file']
    global df
    df = pd.read_csv(csv_file)

    dropable_columns=["id", "ID", "Id", "iD"]

    for i in range(len(dropable_columns)):
        if dropable_columns[i] in df.columns:
            df.drop(dropable_columns[i], inplace=True, axis=1)

    col_names= df.columns

    return render_template("select_target.html", title="Target variable determination", col_names=col_names)




@app.route("/data_exploratory", methods=["GET", "POST"])
def data_exploratory():
    global target
    target = request.form['selected-value']
    global numerical_cols
    global non_numeric_cols
    global df
    problem_type = ""



    ### Checking for classification or regression?
    target_dtype = df[target].dtype
    if target_dtype == 'object' or df[target].nunique() <= 10:
            problem_type =problem_type +  'classification'
    else:
        problem_type = problem_type + 'regression'
    #
    #
    ### Running respective task in background
    # if problem_type == 'classification':
    #     threading.Thread(target=ml_classification_task).start()
    # else:
    #     threading.Thread(target=ml_regression_task).start()



    ### seperating the variables
    for column in df.columns:
            if np.issubdtype(df[column].dtype, np.number):
                numerical_cols.append(column)
            else:
                non_numeric_cols.append(column)




    no_of_obs, no_of_variables = df.shape

    global col_names
    col_names= list(df.columns)

    total_missing_values = df.isna().sum().sum()
    total_missing_values_percentage = (((total_missing_values)/(no_of_obs*no_of_variables))*100).round(1)

    #data_size
    data_size = (df.memory_usage(deep=True).sum()/1024).round(2)

    #Duplicate rows
    duplicate_rows = df.duplicated().sum()
    duplicate_rows_percentage = ((duplicate_rows/no_of_obs)*100).round(1)


    session_values ={"col_names":col_names,
                     "numerical_cols":numerical_cols,
                     "non_numeric_cols":non_numeric_cols,
                     "problem_type":problem_type }

    session["session_values"] = session_values


    return render_template('exploratory.html', no_of_variables=no_of_variables,\
                                               no_of_obs=no_of_obs, col_names=col_names,\
                                                numerical_cols=numerical_cols, \
                                                non_numeric_cols=non_numeric_cols,\
                                                 total_missing_values=total_missing_values,\
                                                 total_missing_values_percentage=total_missing_values_percentage,\
                                                 data_size=data_size,\
                                                 duplicate_rows=duplicate_rows,\
                                                 duplicate_rows_percentage=duplicate_rows_percentage)




@app.route("/feature_data", methods=["POST"])
def feature_data():
    numerical_display_titles = ["Column Type", "Column Range", "Column Mean", "Standard deviation", "Number of missing Values", "Missing Value percentage"]
    non_numerical_display_titles = ["Column Type","Frequent Data", "Number of distinct values", "Percentage of distinct values","Number of missing Values", "Missing Value percentage" ]


    session_values = session.get('session_values')
    if session_values:
        col_names=session_values.get("col_names")
        numerical_cols=session_values.get("numerical_cols")
        non_numeric_cols=session_values.get("non_numeric_cols")
        problem_type = session_values.get("problem_type")

    data_info = {}

    for col in col_names:
        data = df[col]

        # type
        if col in numerical_cols:
            col_type = "Numerical"
            # range
            col_range = ""
            col_range = col_range + str(int(df[col].min())) + "-" + str(int(df[col].max()))
            # mean
            col_mean = (df[col].mean()).round(2)
            #std
            col_std = (df[col].std()).round(2)
            #missing data
            missing_size = df[col].isna().sum()
            missing_size_percantage = ((missing_size/df.shape[0])*100).round(2)
            #graph

            if problem_type == 'classification':
                graph = px.histogram(df, x =df[col], color=target,  template="simple_white")
            else:
                graph = px.histogram(df, x =df[col],  template="simple_white")

            graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)


            data_info[col] = [col_type, col_range, col_mean, col_std, missing_size, missing_size_percantage,graphJSON]
        else:
            col_type = "Non numerical"
            #mode
            col_mode = (df[col].mode()[0])
            # distinct
            distinct_values = df[col].nunique()
            # distinct Percentage
            distinct_values_percentage = str(round((distinct_values/df.shape[1])*100, 2)) + "%"
            #missing data
            missing_size = df[col].isna().sum()
            missing_size_percantage = ((missing_size/df.shape[0])*100).round(2)
            #graph
            if problem_type == 'classification':
                graph = px.histogram(df, x =df[col], color=target,  template="simple_white")
            else:
                graph = px.histogram(df, x =df[col],  template="simple_white")

            graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)

            data_info[col] = [col_type, col_mode, distinct_values, distinct_values_percentage, missing_size, missing_size_percantage, graphJSON]

    return render_template("feature_data.html", data_info=data_info, numerical_display_titles=numerical_display_titles, non_numerical_display_titles=non_numerical_display_titles)



@app.route("/multivariate", methods=["GET", "POST"])
def multivariate():

    # Covariance measures the extent to which two variables vary together.
    # Correlation measures the strength and direction of the linear relationship between two variables.
    correlation = df.corr()
    corr_fig = go.Figure(data=go.Heatmap(
    z=correlation.values,
    x=correlation.columns,
    y=correlation.index,
    colorscale='Magma',
    text=correlation.values.round(2)))
    corr_figJSON = json.dumps(corr_fig, cls=plotly.utils.PlotlyJSONEncoder)

    covariance = df.cov()
    cov_fig = go.Figure(data=go.Heatmap(
    z=covariance.values,
    x=covariance.columns,
    y=covariance.index,
    colorscale='Magma',
    text=covariance.values.round(2)))
    cov_figJSON = json.dumps(cov_fig, cls=plotly.utils.PlotlyJSONEncoder)

    graph_types = ["Scatter", "Line", "Bar"]
    # numeric_df = df[numerical_cols]

    if request.method == 'POST':
        column1 = request.form.get('column1')
        column2 = request.form.get('column2')
        graphType = request.form.get('graphType')

    try:
    # Try plotting the data as-is
        if graphType == "Scatter":
            fig = px.scatter(df, x=column1, y=column2)
            figJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        elif graphType == "Line":
            fig = px.line(df, x=column1, y=column2)
            figJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            fig = px.bar(df, x=column1, y=column2)
            figJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        msg = None
    except (ValueError, TypeError):
        fig = None
        figJSON = None

    return render_template("multivariate.html",cov_figJSON=cov_figJSON,\
                                                corr_figJSON=corr_figJSON,\
                                                correlation=correlation,\
                                                column1=column1,\
                                                 column2=column2,\
                                                  graph_types=graph_types,\
                                                  col_names = col_names,\
                                                  figJSON=figJSON)



@app.route("/model_recommendation", methods=["GET", "POST"])
def model_recommendatio():
    global best
    ## result = best_model
    session_values = session.get('session_values')
    if session_values:
        problem_type=session_values.get("problem_type")

    if problem_type == 'classification':
        from pycaret.classification import setup,  compare_models, pull, evaluate_model, save_model
        s = setup(data = df, target = target)
        best = s.compare_models(budget_time = 0.5)
        table = s.pull()
        table_head = list(table.columns)
        table_data = table.iloc[0].tolist()
        save_model(best, "saved_model")
    else:
        from pycaret.regression import setup, compare_models, pull, evaluate_model, save_model
        s = setup(data = df, target = target)
        best = s.compare_models(budget_time = 0.5)
        table =s. pull()
        table_head = list(table.columns)
        table_data = table.iloc[0].tolist()
        save_model(best, "saved_model")

    return render_template("model_recommendation.html", best = best, table_head=table_head, table_data=table_data, )



@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    session_values = session.get('session_values')
    if session_values:
        col_names=session_values.get("col_names")

        col_names.remove(target)
    return render_template("prediction.html",col_names=col_names)


@app.route("/prediction2", methods=["GET", "POST"])
def prediction2():

    session_values = session.get('session_values')
    if session_values:
        col_names=session_values.get("col_names")
        problem_type= session.get("problem_type")


    col_names.remove(target)
    new_values = []
    pred_sent = False

    for col in col_names:
        value = request.form[col]
        new_values.append(value)


    pred_data_csv = [col_names, new_values]
    pred_data = pd.DataFrame(pred_data_csv[1:], columns=pred_data_csv[0])


    if problem_type=="classification":
        from pycaret.classification import load_model, predict_model
        model = load_model("saved_model")
        final_prediction = predict_model(model, data = pred_data)
        pred_sent = True
    else:
        from pycaret.regression import load_model, predict_model
        model = load_model("saved_model")
        final_prediction = predict_model(model, data = pred_data)
        pred_sent = True

    final_prediction_headers = final_prediction.columns.tolist()
    final_prediction_values = final_prediction.values[0].tolist()

    return render_template("prediction.html", pred_sent=pred_sent,\
                                              col_names=col_names,\
                                              final_prediction_headers=final_prediction_headers,\
                                               final_prediction_values=final_prediction_values)
