import pickledb

if __name__ == "__main__":
    db = pickledb.load("index.db", False)
    db.set("arima", "0")
    #db.set("arimax", "0")
    #db.set("sarima", "13")
    #db.set("sarimax", "10")
    db.dump()