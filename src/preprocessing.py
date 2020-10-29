from read_data import read_from_gs
import pandas as pd
import re


def clean_description(text):
    text = re.sub('[Mm]ag[Cc]hannel', 'Magical Channel', text)
    text = re.sub('\:dfDodge\:','Dodge', text)
    text = re.sub('[Ee]rrata update', '', text)
    text = re.sub('-','minus ', text)
    text = re.sub('\*','',text)
    text = re.sub('\\[tnr]','',text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)

    return(text)

def clean_cards(data):
    data['Description_clean'] = data['Description_clean'].apply(str)
    data['Name'] = data['Name'].apply(str)
    data = data[['Number','Name','Type','Faction','Description_clean']]
    data = data.drop_duplicates(subset = 'Name')
    data['Description_clean'] = [clean_description(name).lower() for name in data['Description_clean']]

    data = data[data['Description_clean'] != 'nan']
    data = data[data['Name'] != 'nan']
    data['Name'] = [name.lower() for name in data['Name']]
    data = data.dropna(0)
    return(data)

def read_and_clean_cards():
    data = read_cards()
    data.rename(columns = {'Description':'Description_clean'}, inplace = True)
    return(clean_cards(data))

def get_id_card_name_dic(cards):
    dic = dict(zip(cards['Number'], cards['Name']))
    return(dic)  

def get_card_name_id_dic(cards):
    dic = dict(zip(cards['Name'], cards['Number']))
    return(dic)

def read_and_clean_validation_set(path, sheet_name):
    df = read_from_gs(path, sheet_name)
    data = pd.melt(df, id_vars = ["Name"])
    data = data[data['variable'] != 'Description']
    data = data[['Name', 'value']]
    data = data[data['value'] != '']

    data = data.dropna(0)
    data['Name'] = [name.lower() for name in data['Name']]
    data['value'] = [name.lower() for name in data['value']]

    data2 = data.copy()
    data2.rename(columns = {'Name':'value', 'value':'Name'}, inplace = True)
    data2.shape

    data3 = pd.concat([data2, data])
    data3 = data3.drop_duplicates()

    data_num_by_name = data3.groupby('Name').count()
    data_num_by_name.rename(columns = {'value':'num'}, inplace = True)
    data4 = pd.merge(data3, data_num_by_name, on = "Name")
    data4 = data4[['Name','value','num']]

    return(data4)


def get_card_faction(number, cards):
    return cards.loc[list(cards['Number'] == number).index(True),'Faction']   

def remove_factions_not_universal_and_not_equal_to(faction, cards):
    cardsToKeep = cards[cards['Faction'].isin([faction,'Universal'])]['Number']
    return { key: deck[key] for key in cardsToKeep }

def get_card_number(card_name):
    for key, value in id_card_name.items():
        if str(value) == card_name:
        return(key)

def deck_decorator(card_number, cards):
    return(cards[ cards['Number'].isin(card_number)])

def get_similar_card(cardName):
    id = get_card_number(cardName.lower())
    cardEmbedding = deck.get(id)
    findCloserCards = closer_distance_dispatcher[CLOSER_POINTS_ALG]

    if cardEmbedding is not None:
        res = findCloserCards(id)
    else :
        res = 'Card ' + cardName + ' Not Found '
        #remove_factions_not_universal_and_not_equal_to(get_card_faction(number, cards), cards)
        return(res)    

def decorate_with_description_and_card_name(df):
    data1 = df.join(cards.set_index('Number'), on = 'Number', how = 'left')
    data1 = data1[['card_name', 'Name', 'Description_clean']]
    data1.rename(columns = {'Name':'similar_card_name', 'Description_clean':'similar_card_description_clean'}, inplace = True)
    data1 = data1.join(cards.set_index('Name'), on = 'card_name', how = 'left')
    data1 = data1[['card_name', 'Description_clean', 'similar_card_name', 'similar_card_description_clean']]
    data1.rename(columns = {'Description_clean':'description_clean'}, inplace = True)
    return(data1)
