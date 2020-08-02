
def remove_white_spaces_and_special_characters(sentence):
  nowhitespaces = re.sub('[^A-Za-z0-9 ]+', '', sentence)
  return(nowhitespaces)

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

def read_from_gs(path, sheet_name):
  from google.colab import auth
  auth.authenticate_user()

  import gspread
  from oauth2client.client import GoogleCredentials

  gc = gspread.authorize(GoogleCredentials.get_application_default())
  wb = gc.open_by_url(path)
  sheet = wb.worksheet(sheet_name)

  records = sheet.get_all_values()
  data = pd.DataFrame.from_records(records[1:])
  dic_rename = dict(zip(data.columns.tolist(), records[0]))
  data.rename(columns = dic_rename, inplace = True)
  
  return(data)

def read_and_clean_gs(path, sheet_name):
  
  data = read_from_gs(path, sheet_name)
  data.rename(columns = {'Description':'Description_clean'}, inplace = True)
  return(clean_cards(data))


def read_and_clean_csv(path):
  df = pd.read_csv(path)
  df.rename(columns= {'Description.1':'Description_clean'}, inplace = True)
  return(clean_cards(df))

def get_id_card_name_dic(cards):
  dic = dict(zip(cards['Number'], cards['Name']))
  return(dic)  

def get_card_name_id_dic(cards):
  dic = dict(zip(cards['Name'], cards['Number']))
  return(dic)

def read_and_clean_validation_set(path, sheet_name):
  df = read_from_gs(path, sheet_name)
  data = pd.melt(df, id_vars=["Name"])
  data = data[data['variable'] != 'Description']
  data = data[['Name', 'value']]
  data = data[data['value'] != '']

  data = data.dropna(0)
  data[data['Name'] == 'Combination Strike']
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
