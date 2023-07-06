def my_function(x, y):
    res = x * y
    if res >= 0:
        if res == 0:
            pass  # Do nothing
        else:
            if res % 2 == 0:
                res = res / 2
            res = res * 10
    else:
        res = -1
    # drucke res und x und y
    print(res)
    print(x)
    print(y)
    return res


my_function(1, 1)


# READ IN CSV WITH DATA FOR EACH DAY AND EXTEND TO HOURLY DATA'
df_isar = load_data("Zeitreihen/Isartemperatur.csv")
new_index = np.repeat(df_isar.index.values, 24)
df_hourly = df_isar.reindex(new_index).ffill()
df_hourly.reset_index(drop=True, inplace=True)
df_Zeitreihen["Isartemp"] = df_hourly["Isartemp"]
df_hourly.to_csv("filename.csv", sep=";", decimal=",", index=False)
