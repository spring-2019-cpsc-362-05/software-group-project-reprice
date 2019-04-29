import pandas as pd

def main():
    df = pd.read_csv('data/Neighborhood_MedianValuePerSqft_AllHomes.csv')
    df = df['RegionName'].values

    file_obj = open("nb.txt", 'w')
    for elem in df:
        file_obj.write("\"{}\",".format(elem))

    file_obj.close()


    # for row in df.

    # state_df = pd.read_csv('data/State_MedianValuePerSqft_AllHomes.csv')
    # df = pd.read_csv("http://files.zillowstatic.com/research/public/State/State_MedianValuePerSqft_AllHomes.csv")
    # print(df.head())
    # state_df = state_df['RegionName']
    # state_df.to_csv("valid_entries/state_entries.csv", index=False)
    #
    # city_df = pd.read_csv("data/City_MedianValuePerSqft_AllHomes.csv", encoding='latin-1')
    # city_df = city_df['RegionName']
    # city_df.to_csv("valid_entries/city_entries.csv", index=False)
    #
    # county_df = pd.read_csv('data/County_MedianValuePerSqft_AllHomes.csv', encoding='latin-1')
    # county_df = county_df['RegionName']
    # county_df.to_csv("valid_entries/county_entries.csv", index=False)
    #
    # neighborhodd_df = pd.read_csv('data/Neighborhood_MedianValuePerSqft_AllHomes.csv')
    # neighborhodd_df = neighborhodd_df['RegionName']
    # neighborhodd_df.to_csv("valid_entries/neighborhood_entries.csv", index=False)


if __name__ == '__main__': main()
