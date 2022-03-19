from tkinter.ttk import Style
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.close("all")
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

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


def listed_company_per_year():
    stock_basic_df = pd.read_csv("data/stock_basic.csv")
    stock_basic_df["list_year"] = stock_basic_df["list_date"]//10000
    stock_basic_list_year_group = stock_basic_df.groupby("list_year").count()
    stock_basic_list_year_group["list_year"] = list(stock_basic_list_year_group.index)
    plt.figure()
    plt.title("list company count per year")
    stock_basic_list_year_group.plot(y="ts_code")
    
    plt.savefig("data/list_company_count_per_year.jpg")
    
    stock_basic_list_year_group["count"] = stock_basic_list_year_group["ts_code"]
    stock_basic_list_year_group.loc[:,"count"].to_csv("data/list_company_count_per_year.csv")

def listed_company_per_area():
    df = pd.read_csv("data/stock_basic.csv")
    df_groupby_area = df.groupby("area")
    df_area_count = df_groupby_area.count()
    plt.figure()
    plt.title("list company count per area")
    df_area_count.plot(y="ts_code",kind="bar")
    plt.savefig("data/list_company_count_per_area.jpg")
    df_area_count["count"] = df_area_count["ts_code"]
    df_area_count.loc[:,"count"].to_csv("data/list_company_count_per_area.csv")


def listed_company_per_industry():
    df = pd.read_csv("data/stock_basic.csv")
    df_groupby_industry = df.groupby("industry")
    df_industry_count = df_groupby_industry.count()
    plt.figure()
    plt.title("list company count per industry")
    df_industry_count.plot(y="ts_code",kind="bar")
    plt.savefig("data/list_company_count_per_industry.jpg")
    df_industry_count["count"] = df_industry_count["ts_code"]
    df_industry_count.loc[:,"count"].to_csv("data/list_company_count_per_industry.csv")
    
def industry_company(industry):
    df = pd.read_csv("data/stock_basic.csv")
    company_df = df[df["industry"]==industry]
    company_df.to_csv("data/{}_company.csv".format(industry))
    
def industry_per_area():
    df = pd.read_csv("data/stock_basic.csv")
    df_area_industry = df.groupby(["area","industry"]).count()
    df_area_industry["count"] = df_area_industry["ts_code"]
    df_area_industry.loc[:,"count"].to_csv("data/industry_per_area.csv")
    
if __name__ == "__main__":
    daily_organized_root = "data/origin_data/date_organized"
    company_organized_root = "data/origin_data/company_organized"
    # company_numbers_per_year(daily_organized_root)
    # listed_company_per_year()
    # listed_company_per_area()
    # listed_company_per_industry()
    # industry_company("文教休闲")
    industry_per_area()