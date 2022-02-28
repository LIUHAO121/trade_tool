from tkinter.ttk import Style
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.close("all")


def company_numbers_per_year(daily_organized_root):
    file_path = "data/basic/company_num.csv"
    if not os.path.exists(file_path):
        date_file_names = os.listdir(daily_organized_root)
        
        df_list = []
        
        for date_file in date_file_names:
            date = date_file[:-4]
            daily_df = pd.read_csv(os.path.join(daily_organized_root,date_file))
            num_company = len(daily_df)
            pair = [date,num_company]
            df_list.append(pair)
        df = pd.DataFrame(df_list,columns=["date","company_num"])
        df_sort = df.sort_values(by="date",ascending=True).reset_index(drop=True)
        df_sort.to_csv(file_path)
    else:
        plt.figure()
        df = pd.read_csv(file_path)
        df.plot(x="date",y="company_num")
        plt.savefig("data/his_company_num.png",
            dpi=1000,bbox_inches = 'tight')


if __name__ == "__main__":
    daily_organized_root = "data/origin_data/date_organized"
    company_organized_root = "data/origin_data/company_organized"
    company_numbers_per_year(daily_organized_root)
    