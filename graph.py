import pandas as pd

def graph_aggregate_eval_metrics():
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    def get_data():
        import db_connect

        conn = db_connect.establish_connection()
        cursor = conn.cursor()

        Q1 = """select * from basic_X_MA_crossover_V1 where Strategy_ID <= 8600"""

        x = cursor.execute(Q1)
        rows = x.fetchall()

        conn.commit()
        conn.close()

        return rows

    def sort_data(data):

        data.sort(key=lambda y:y[8],reverse=True)

        new_data = []
        for enum,i in enumerate(data):
            tl = list(i)
            tl[0] = enum
            i = tuple(tl)
            new_data.append(i)

        return new_data

    def normalize_param_vals(data):

        data_df = pd.DataFrame(data)
        data_df = data_df[[0,4,8]]



        data_df.rename(columns={0:'trade num',8:'total R profit'},inplace=True)

        # convert string tuple to list
        data_df[4] = [list(eval(r)) for r in data_df[4].to_numpy()]

        # split the param tuple up
        for i in range(len(data_df.iloc[0][4])):
            data_df[i] = data_df[4].str[i]
            data_df.rename(columns={i: 'param '+str(i+1)}, inplace=True)

        data_df.drop(4,axis=1,inplace=True)

        data_df = data_df.set_index('trade num')
        data_df = data_df.apply(lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)))
        data_df.reset_index(inplace=True)

        return data_df

    ##########################################################

    rows = get_data()
    rows = sort_data(data=rows)
    rows = normalize_param_vals(data=rows)
    x = []
    y = []

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows['trade num'], y=rows['total R profit'], mode='lines', name='Total Realized R', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=rows['trade num'], y=rows['param 1'], mode='lines', name='param 1',line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=rows['trade num'], y=rows['param 2'], mode='lines', name='param 2', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=rows['trade num'], y=rows['param 3'], mode='lines', name='param 3', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=rows['trade num'], y=rows['param 4'], mode='lines', name='param 4', line=dict(color='black')))

    fig.show()