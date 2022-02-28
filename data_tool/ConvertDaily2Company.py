import os
import pandas as pd

def get_all_date(daily_organized_root):
    file_names = os.listdir(daily_organized_root)
    dates = []
    for name in file_names:
        if ".csv" in name:
            date = name[:-4]
            dates.append(date)
    return dates

def get_all_company_code(daily_organized_root):
    one_daily_name = os.listdir(daily_organized_root)[0]
    daily_file = os.path.join(daily_organized_root,one_daily_name)
    daily_df = pd.read_csv(daily_file)
    company_codes = list(daily_df["ts_code"].values)
    return company_codes
    

def get_history_data_by_tscode(ts_code,daily_organized_root,company_organized_root):
    daily_files = os.listdir(daily_organized_root)
    columns = list(pd.read_csv(os.path.join(daily_organized_root,daily_files[0])).columns)[1:] # 去掉索引
    all_row_values = []
    for daily_f in daily_files:
        daily_df = pd.read_csv(os.path.join(daily_organized_root,daily_f))
        row,col = daily_df.shape

        code_df = daily_df[daily_df["ts_code"]==ts_code]
        if len(code_df) >= 1:
            code_row = code_df.iloc[0]
            code_row_values = list(code_row.values)[1:]  # 去掉索引
            all_row_values.append(code_row_values)
    
    df = pd.DataFrame(all_row_values,columns=columns)
    out_path = os.path.join(company_organized_root,"{}.csv".format(ts_code))
    df_datesort = df.sort_values(by="trade_date",ascending=False).reset_index(drop=True)
    df_datesort.to_csv(out_path)
        
        


if __name__ == "__main__":
    daily_organized_root = "data/origin_data/date_organized"
    company_organized_root = "data/origin_data/company_organized"
    
    all_dates = get_all_date(daily_organized_root=daily_organized_root)
    st_codes = get_all_company_code(daily_organized_root=daily_organized_root)
    for st_code in st_codes:
        get_history_data_by_tscode(st_code,daily_organized_root,company_organized_root)
    