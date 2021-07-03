from flask import Flask, render_template
from flask import request
import pandas as pd
import pickle
from visualization import X_all
soccer = Flask(__name__)


clf = pickle.load(open('tunedmodel','rb'))
@soccer.route('/')
def runpy():
    return render_template('index.html')

@soccer.route('/',methods=['POST'])
def form_post():
    global team1
    global team2
    team1 = request.form['sel1']
    team2 = request.form['sel2']
    if team1 != team2 :
        teamname = team1
        data2 = pd.read_csv('team_dataframe.csv')
        data2 = data2.iloc[30:]
        global teamindex
        teamindex = 122
        for index, row in data2.iterrows():
            if teamname == row['HomeTeam']:
                teamindex = index
        winnerlist = clf.predict(X_all.loc[teamindex].to_frame().T)
        print(winnerlist)
        global teamwin
        global hnh
        teamwin = winnerlist[0]
        if teamwin == 'A':
            teamwin = team2
            hnh = "Away Team Wins"
        elif teamwin == 'H':
            teamwin = team1
            hnh = "Home Team Wins"
        else:
            teamwin = "DRAW!"
            hnh = "The game will be a DRAW"
        print(teamwin)
    else:
        return render_template('index.html', text='None',blehh='\tPlease Select Different teams.')
    return render_template('index.html', text=teamwin,bleh=team2,blehh=hnh)



if __name__ == '__main__':
    soccer.debug = True
    soccer.run()
    soccer.run(debug=True)
