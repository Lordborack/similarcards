#from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials
import pandas as pd
import config


def read_from_gs(path, sheet_name):
    #auth.authenticate_user()

    gc = gspread.authorize(GoogleCredentials.get_application_default())
    wb = gc.open_by_url(path)
    sheet = wb.worksheet(sheet_name)

    records = sheet.get_all_values()
    data = pd.DataFrame.from_records(records[1:])
    dic_rename = dict(zip(data.columns.tolist(), records[0]))
    data.rename(columns = dic_rename, inplace = True)
    return(data)


def read_cards():
    path = construct_input_data_path(config.CARDS_FILE_NAME)
    return(pd.read_csv(path))


def construct_input_data_path(fileName):
    return(config.INPUT_PATH + fileName)


