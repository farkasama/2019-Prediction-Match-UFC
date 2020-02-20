#encode=utf-8

import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

""" PARAMETRES POUR LES GRAPHES
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio

importances = None
X = None
"""

#Fonction qui converti les livres en kilogrammes
def lbsToKg(poids):
    if poids != poids:
        return None
    return round(float(poids)*0.453592, 2)

#Fonction qui converti dexu chiffres en pourcentage
def toPct(v1, v2):
    if v2 == 0:
        return 0
    return v1/v2



def read_data(data):

    #Cette partie recupere juste les combats qui sont superieurs a 1999
    data['year'] = data['date'].apply(lambda x : x.split('-')[0])
    data['year'] = data['year'].astype(int)
    data = data[data['year'] > 1999]
    data = data.sort_values(by=['date'], ascending=False)

    #Creation des disctionnaires
    fighter = dict()
    nbout = 0
    fighter_out = dict()

    #Pour toutes les lignes du data
    for i in range(len(data)):

        #On recupere le nom du combattant
        B_fighter = data.at[i, 'B_fighter']
        
        #Si il n'est pas deja dans le dictionnaire, alors on l'ajoute avec ses informations (on ne prends donc que son dernier match)
        if B_fighter not in fighter:
            B_age = data.at[i, 'B_age']
            B_height = data.at[i, 'B_Height_cms']
            B_reach = data.at[i, 'B_Reach_cms']
            B_weight = lbsToKg(data.at[i, 'B_Weight_lbs'])
            B_wins = data.at[i, 'B_wins']
            B_losses = data.at[i, 'B_losses']
            B_total_time = data.at[i, 'B_total_time_fought(seconds)']
            B_total_title_bouts = data.at[i, 'B_total_title_bouts']
            B_streak = data.at[i, 'B_current_win_streak'] + (-1)*data.at[i, 'B_current_lose_streak']
            B_longest_win_streak = data.at[i, 'B_longest_win_streak']
            B_body = toPct(data.at[i, 'B_avg_BODY_landed'], data.at[i, 'B_avg_BODY_att'])
            B_clinch = toPct(data.at[i, 'B_avg_CLINCH_landed'], data.at[i, 'B_avg_CLINCH_att'])
            B_distance = toPct(data.at[i, 'B_avg_DISTANCE_landed'], data.at[i, 'B_avg_DISTANCE_att'])
            B_ground = toPct(data.at[i, 'B_avg_GROUND_landed'], data.at[i, 'B_avg_GROUND_att'])
            B_head = toPct(data.at[i, 'B_avg_HEAD_landed'], data.at[i, 'B_avg_HEAD_att'])
            B_kd = data.at[i, 'B_avg_KD']
            B_leg = toPct(data.at[i, 'B_avg_LEG_landed'], data.at[i, 'B_avg_LEG_att'])
            B_pass = data.at[i, 'B_avg_PASS']
            B_rev = data.at[i, 'B_avg_REV']
            B_sig_str = data.at[i, 'B_avg_SIG_STR_pct']
            B_sub = data.at[i, 'B_avg_SUB_ATT']
            B_td = data.at[i, 'B_avg_TD_pct']
            B_total_str = toPct(data.at[i, 'B_avg_TOTAL_STR_landed'], data.at[i, 'B_avg_TOTAL_STR_att'])
            B_draw = data.at[i, 'B_draw']
            b_pourcent = toPct(float(B_wins), float(B_wins)+float(B_losses)+float(B_draw))

            fb = [B_age, B_height, B_reach, B_weight, B_wins, B_losses, B_total_time, B_total_title_bouts, B_streak, B_longest_win_streak, B_body, B_clinch, B_distance, B_ground, B_head, B_kd, B_leg, B_pass, B_rev, B_sig_str, B_sub, B_td, B_total_str, b_pourcent]

            #Si une des informations est un NaN, alors nous n'ajoutons pas le combattant dans notre dictionnaire
            b = True
            for j in range(len(fb)):
                if fb[j] != fb[j]:
                    b = False
                    if B_fighter not in fighter_out:
                        nbout = nbout+1
                        fighter_out[B_fighter] = True
                    break
            if (b):
                fighter[B_fighter] = fb


        #On recupere le nom du combattant
        R_fighter = data.at[i, 'R_fighter']
        
        #Si il n'est pas deja dans le dictionnaire, alors on l'ajoute avec ses informations (on ne prends donc que son dernier match)
        if R_fighter not in fighter:
            R_age = data.at[i, 'R_age']
            R_height = data.at[i, 'R_Height_cms']
            R_reach = data.at[i, 'R_Reach_cms']
            R_weight = lbsToKg(data.at[i, 'R_Weight_lbs'])
            R_wins = data.at[i, 'R_wins']
            R_losses = data.at[i, 'R_losses']
            R_total_time = data.at[i, 'R_total_time_fought(seconds)']
            R_total_title_bouts = data.at[i, 'R_total_title_bouts']
            R_streak = data.at[i, 'R_current_win_streak'] + (-1)*data.at[i, 'R_current_lose_streak']
            R_longest_win_streak = data.at[i, 'R_longest_win_streak']
            R_body = toPct(data.at[i, 'R_avg_BODY_landed'], data.at[i, 'R_avg_BODY_att'])
            R_clinch = toPct(data.at[i, 'R_avg_CLINCH_landed'], data.at[i, 'R_avg_CLINCH_att'])
            R_distance = toPct(data.at[i, 'R_avg_DISTANCE_landed'], data.at[i, 'R_avg_DISTANCE_att'])
            R_ground = toPct(data.at[i, 'R_avg_GROUND_landed'], data.at[i, 'R_avg_GROUND_att'])
            R_head = toPct(data.at[i, 'R_avg_HEAD_landed'], data.at[i, 'R_avg_HEAD_att'])
            R_kd = data.at[i, 'R_avg_KD']
            R_leg = toPct(data.at[i, 'R_avg_LEG_landed'], data.at[i, 'R_avg_LEG_att'])
            R_pass = data.at[i, 'R_avg_PASS']
            R_rev = data.at[i, 'R_avg_REV']
            R_sig_str = data.at[i, 'R_avg_SIG_STR_pct']
            R_sub = data.at[i, 'R_avg_SUB_ATT']
            R_td = data.at[i, 'R_avg_TD_pct']
            R_total_str = toPct(data.at[i, 'R_avg_TOTAL_STR_landed'], data.at[i, 'R_avg_TOTAL_STR_att'])
            R_draw = data.at[i, 'R_draw']
            r_pourcent = toPct(float(R_wins), float(R_wins)+float(R_losses)+float(R_draw))

            fr = [R_age, R_height, R_reach, R_weight, R_wins, R_losses, R_total_time, R_total_title_bouts, R_streak, R_longest_win_streak, R_body, R_clinch, R_distance, R_ground, R_head, R_kd, R_leg, R_pass, R_rev, R_sig_str, R_sub, R_td, R_total_str, r_pourcent]
            
            #Si une des informations est un NaN, alors nous n'ajoutons pas le combattant dans notre dictionnaire
            b = True
            for j in range(len(fr)):
                if fr[j] != fr[j]:
                    b = False
                    if R_fighter not in fighter_out:
                        nbout = nbout+1
                        fighter_out[R_fighter] = True
                    break
            if b:
                fighter[R_fighter] = fr
    

    print("Modifications du dataset : tous les matchs sont a partir de l'annee 2000")
    print("Nombre de match : " + str(len(data)))
    print("Nombre de fighter dans notre dataset : "+ str(len(fighter)+nbout))
    print("Nombre de fighter acceptable : " + str(len(fighter)))
    print("Nombre de fighter ou il manque des donnees : " + str(nbout))
    return fighter, data


#Retourne un tableau qui ajoute a sa case i la soustraction fighter1[i] - fighter2[i]
def compareFighter(fighter1, fighter2):
    res = []
    for i in range(len(fighter1)):
        res.append(fighter1[i] - fighter2[i])
    return res
    

#Retourne un ensemble de train et de test
def trainData(data, fighter, n, seed):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    i = 0
    random.seed(seed)
    while i < n:
        #On verifie que les combattants sont bien dans le dictionnaire
        if data.at[i, 'R_fighter'] in fighter and data.at[i, 'B_fighter'] in fighter:
            #Si le random est inferieur a 0.3 on ajoute dans le test, sinon dans le train
            if random.random() < 0.3:
                test_x.append(compareFighter(fighter[data.at[i, 'R_fighter']], fighter[data.at[i, 'B_fighter']]))
                test_y.append(data.at[i, 'Winner'])
            else:
                train_x.append(compareFighter(fighter[data.at[i, 'R_fighter']], fighter[data.at[i, 'B_fighter']]))
                train_y.append(data.at[i, 'Winner'])
        i = i+1
    return train_x, train_y, test_x, test_y


#Predit le gagnant entre le fighter1 et le fighter2 en fonction du dataset filename
def predictAFight(fighter1, fighter2, filename):

    #On ouvre le .csv avec la librairie pandas
    data = pd.read_csv(filename)

    #On recupere le dictionnaire et les modifications du data avec notre fonction
    fighter, data = read_data(data)

    #Si le fighter1 ou le fighter2 ne sont pas dans notre dictionnaire, on ne peut donc pas predire
    if fighter1 not in fighter:
        print("Erreur sur le nom %s : il n'existe pas dans la base de donnees." % fighter1)
        return
    if fighter2 not in fighter:
        print("Erreur sur le nom %s : il n'existe pas dans la base de donnees." % fighter2)
        return

    #On recupere nos ensembles grace a la fonction plus haut
    train_x, train_y, test_x, test_y = trainData(data, fighter, 3500, 10)

    #Creation du Random Forest
    clf = RandomForestClassifier(max_depth=5, n_estimators=1000)

    #On lui ajoute les trains
    clf.fit(train_x, train_y)

    #On recupere l'ensemble de prediction
    prediction = clf.predict(test_x)

    """ PARAMETRES POUR LES GRAPHES
    global importances
    importances = clf.feature_importances_
    global X
    X = np.array(train_x)
    """
    
    #On recupere l'accuracy score
    score = accuracy_score(test_y, prediction)

    #On recupere la prediction ainsi que la probabilité entre le fighter1 et le fighter2
    prediction = clf.predict([compareFighter(fighter[fighter1], fighter[fighter2])])[0]
    p = clf.predict_proba([compareFighter(fighter[fighter1], fighter[fighter2])])

    #On affiche toutes les informations
    print("Score accurency : %5.2f%s" % (round(score, 4) *100, "%"))
    print("Combat entre %s et %s :" % (fighter1, fighter2))
    if prediction == 'Blue':
        print("Le comabatant %s a plus de chances de gagner avec une probabilite de %5.2f%s." % (fighter2, round(p[0][0], 4)*100, "%"))
    else:
        print("Le comabatant %s a plus de chances de gagner avec une probabilite de %5.2f%s." % (fighter1, round(p[0][2], 4)*100, "%"))


#Appelle de la fonction
predictAFight('Conor McGregor', 'Khabib Nurmagomedov', "data.csv")


""" AFFICHAGE GRAPH
indices = np.argsort(importances)[::-1]

feature = ['Age', 'Height', 'Reach', 'Weight', 'Wins', 'Losses', 'Total Time Fought', 'Total Title Bouts', 'Current Streak', 'Longest Win Streak', 'Body', 'Clinch', 'Distance', 'Ground', 'Head', 'Knockdowns', 'Leg', 'Pass', 'Reversals', 'Significant Strikes', 'Submission Attempts', 'Takedowns', 'Total Strikes']
features = []

for i in range(X.shape[1]):
    features.append(feature[indices[i]])

trace = go.Bar(y=importances[indices], x=features)
layout = go.Layout(title='Importances features', 
                   xaxis={'title':'Features'},
                  yaxis={'title' : 'Events'})
fig = go.Figure(data=trace, layout=layout)
fig.show()
"""